using_gurobi = False
using_python_mip = True

from abc import ABC, abstractmethod
from typing import Tuple, List, Union, Generator, Dict
import shutil
import tempfile
import os
from numpy import isclose

if using_python_mip:
    from mip import Model, LinExpr, Var
if using_gurobi:
    import gurobipy as grb


class FileReaderInterface(ABC):

    @abstractmethod
    def get_n_vars(self) -> int:
        pass

    @abstractmethod
    def get_n_cons(self) -> int:
        pass

    @abstractmethod
    def get_nnz(self) -> int:
        pass

    @abstractmethod
    def get_var_bounds(self) -> Tuple[List[float]]:
        pass

    @abstractmethod
    def get_lrhss(self) -> Tuple[List[float]]:
        pass

    @abstractmethod
    def get_cons_matrix(self) -> Tuple[List[Union[int, float]]]:
        pass

    @abstractmethod
    def get_SCIP_vartypes(self) -> List[int]:
        pass

    @abstractmethod
    def write_model_with_new_bounds(self, lbs: List[float], ubs: [List[float]]) -> str:
        pass


class PythonMIPReader(FileReaderInterface):
    def __init__(self, input_file: str) -> None:
        self.instance_name = input_file.split("/")[-1].split(".")[0]
        self.m = Model()

        print("Reding lp file", self.instance_name)
        self.m.read(input_file)
        print("Reading of ", self.instance_name, " model done!")

        self.vars = self.m.vars
        self.constrs = self.m.constrs

        self.tmp_reader_dir = tempfile.mkdtemp()

    def __del__(self):
        shutil.rmtree(self.tmp_reader_dir)


    def write_model_with_new_bounds(self, lbs: List[float], ubs: [List[float]]) -> str:
        assert(len(lbs) == len(ubs) == len(self.m.vars))
        for i in range(len(lbs)):
            self.m.vars[i].lb = lbs[i]
            self.m.vars[i].ub = ubs[i]
        out_path = os.path.join(self.tmp_reader_dir, str(self.instance_name) + ".mps")
        self.m.write(out_path)
        return out_path

    def get_n_vars(self) -> int:
        return self.m.num_cols

    def get_n_cons(self) -> int:
        return self.m.num_rows

    def get_nnz(self) -> int:
        return self.m.num_nz

    def get_var_bounds(self) -> Tuple[List[float]]:
        ubs = map(lambda var: var.ub, self.vars)
        lbs = map(lambda var: var.lb, self.vars)
        return list(lbs), list(ubs)

    def get_lrhss(self) -> Tuple[List[float]]:
        lhss = map(
            lambda cons: float('-Inf') if cons.expr.sense == '<' else cons.rhs,
            self.constrs
        )
        rhss = map(
            lambda cons: float('Inf') if cons.expr.sense == '>' else cons.rhs,
            self.constrs
        )
        return list(lhss), list(rhss)

    def get_cons_matrix(self) -> Tuple[List[Union[int, float]]]:

        def get_expr_coos(expr: LinExpr, var_indices: Dict[Var, int]) -> Generator:
            for var, coeff in expr.expr.items():
                yield coeff, var_indices[var]

        row_indices = []
        row_ptrs = []
        col_indices = []
        coeffs = []

        var_indices = {v: i for i, v in enumerate(self.vars)}

        row_ctr = 0
        row_ptrs.append(row_ctr)

        for row_idx, constr in enumerate(self.constrs):

            for coeff, col_idx in get_expr_coos(constr.expr, var_indices):
                row_ctr += 1
                row_indices.append(row_idx)
                col_indices.append(col_idx)
                coeffs.append(coeff)

            row_ptrs.append(row_ctr)

        return (coeffs, row_ptrs, col_indices)

    def get_SCIP_vartypes(self) -> List[int]:
        conversion_dict = {'B': 0, 'I': 1, 'C': 3}
        python_mip_vartypes = map(lambda var: var.var_type, self.vars)

        # if a variable is binary, assert that the bounds are [0,1]
        for var in self.vars:
            if var.var_type == 'B':
                assert(isclose(var.ub, 1.0))
                assert(isclose(var.lb, 0.0))

        return list(map(
            conversion_dict.get,
            python_mip_vartypes
        ))


class GurobiReader(FileReaderInterface):
    def __init__(self, instance: str) -> None:
        self.m = grb.read(instance)
        self.dvars = self.m.getVars()
        self.constrs = self.m.getConstrs()

    def get_n_vars(self) -> int:
        return self.m.getAttr('NumVars')

    def get_n_cons(self) -> int:
        return self.m.getAttr('NumConstrs')

    def get_nnz(self) -> int:
        return self.m.getAttr('NumNZs')

    def get_var_bounds(self) -> Tuple[List[float]]:
        ubs = self.m.getAttr('UB', self.dvars)
        lbs = self.m.getAttr('LB', self.dvars)
        return lbs, ubs

    def get_cons_matrix(self) -> Tuple[List[Union[int, float]]]:

        def get_expr_coos(expr, var_indices):
            for i in range(expr.size()):
                dvar = expr.getVar(i)
                yield expr.getCoeff(i), var_indices[dvar]

        row_indices = []
        row_ptrs = []
        col_indices = []
        coeffs = []

        var_indices = {v: i for i, v in enumerate(self.dvars)}

        row_ctr = 0
        row_ptrs.append(row_ctr)

        for row_idx, constr in enumerate(self.constrs):

            for coeff, col_idx in get_expr_coos(self.m.getRow(constr), var_indices):
                row_ctr += 1
                row_indices.append(row_idx)
                col_indices.append(col_idx)
                coeffs.append(coeff)

            row_ptrs.append(row_ctr)

        return (coeffs, row_ptrs, col_indices)

    def get_lrhss(self) -> Tuple[List[float]]:
        lhss = map(
            lambda cons: float('-Inf') if cons.getAttr('Sense') == '<' else cons.getAttr('RHS'),
            self.constrs
        )
        rhss = map(
            lambda cons: float('Inf') if cons.getAttr('Sense') == '>' else cons.getAttr('RHS'),
            self.constrs
        )
        return list(lhss), list(rhss)

    def get_SCIP_vartypes(self) -> List[int]:
        conversion_dict = {'B': 0, 'I': 1, 'C': 3}
        gurobi_vartypes = map(lambda var: var.getAttr('VType'), self.dvars)
        return list(map(
            conversion_dict.get,
            gurobi_vartypes
        ))


# Instantiation
def get_reader(input_file: str, reader: str = 'python-mip') -> FileReaderInterface:
    if reader == 'python-mip':
        return PythonMIPReader(input_file)
    elif reader == 'gurobi':
        return GurobiReader(input_file)
    else:
        raise Exception("Unsupported reader: " + reader)
