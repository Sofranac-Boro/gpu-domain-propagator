from ctypes import *
from ctypes import _SimpleCData
from typing import List, Tuple
from enum import Enum

so_file = "../build/libGpuProp.so"


def check_input(
        n_vars: int,
        n_cons: int,
        nnz: int,
        csr_col_indices: List[int],
        csr_row_ptrs: List[int],
        csr_vals: List[float],
        lhss: List[float],
        rhss: List[float],
        lbs: List[float],
        ubs: List[float],
        vartypes: List[int]
):
    assert len(csr_col_indices) == nnz
    assert len(csr_row_ptrs) == n_cons + 1
    assert len(csr_vals) == nnz
    assert len(lhss) == n_cons
    assert len(rhss) == n_cons
    assert len(lbs) == n_vars
    assert len(ubs) == n_vars
    assert len(vartypes) == n_vars


def propagateGPUReduction(
        n_vars: int,
        n_cons: int,
        nnz: int,
        csr_col_indices: List[int],
        csr_row_ptrs: List[int],
        csr_vals: List[float],
        lhss: List[float],
        rhss: List[float],
        lbs: List[float],
        ubs: List[float],
        vartypes: List[int],
        datatype: _SimpleCData = c_double
) -> Tuple[List[float]]:
    C = CDLL(so_file)

    check_input(n_vars, n_cons, nnz, csr_col_indices, csr_row_ptrs, csr_vals, lhss, rhss,
                lbs, ubs, vartypes)

    c_n_vars = c_int(n_vars)
    c_n_cons = c_int(n_cons)
    c_nnz = c_int(nnz)
    c_csr_col_indices = (c_int * nnz)(*csr_col_indices)
    c_csr_row_ptrs = (c_int * (n_cons + 1))(*csr_row_ptrs)
    c_csr_vals = (datatype * nnz)(*csr_vals)
    c_lhss = (datatype * n_cons)(*lhss)
    c_rhss = (datatype * n_cons)(*rhss)
    c_lbs = (datatype * n_vars)(*lbs)
    c_ubs = (datatype * n_vars)(*ubs)
    c_vartypes = (c_int * n_vars)(*vartypes)

    if datatype == c_double:
        fun = C.propagateConstraintsGPUReductionDouble
    elif datatype == c_float:
        fun = C.propagateConstraintsGPUReductionFloat
    else:
        raise Exception("unsupported datatype")

    fun(
        c_n_cons,
        c_n_vars,
        c_nnz,
        c_csr_col_indices,
        c_csr_row_ptrs,
        c_csr_vals,
        c_lhss,
        c_rhss,
        c_lbs,
        c_ubs,
        c_vartypes
    )

    return list(c_lbs), list(c_ubs)


def propagateGPUAtomic(
        n_vars: int,
        n_cons: int,
        nnz: int,
        csr_col_indices: List[int],
        csr_row_ptrs: List[int],
        csr_vals: List[float],
        lhss: List[float],
        rhss: List[float],
        lbs: List[float],
        ubs: List[float],
        vartypes: List[int],
        fullAsync = True,
        datatype = c_double
) -> Tuple[List[float]]:
    C = CDLL(so_file)

    check_input(n_vars, n_cons, nnz, csr_col_indices, csr_row_ptrs, csr_vals, lhss, rhss,
                lbs, ubs, vartypes)

    c_n_vars = c_int(n_vars)
    c_n_cons = c_int(n_cons)
    c_nnz = c_int(nnz)
    c_csr_col_indices = (c_int * nnz)(*csr_col_indices)
    c_csr_row_ptrs = (c_int * (n_cons + 1))(*csr_row_ptrs)
    c_csr_vals = (datatype * nnz)(*csr_vals)
    c_lhss = (datatype * n_cons)(*lhss)
    c_rhss = (datatype * n_cons)(*rhss)
    c_lbs = (datatype * n_vars)(*lbs)
    c_ubs = (datatype * n_vars)(*ubs)
    c_vartypes = (c_int * n_vars)(*vartypes)
    c_fullAsync = (c_bool)(fullAsync)

    if datatype == c_double:
        fun = C.propagateConstraintsGPUAtomicDouble
    elif datatype == c_float:
        fun = C.propagateConstraintsGPUAtomicFloat
    else:
        raise Exception("unsupported datatype")

    fun(
        c_n_cons,
        c_n_vars,
        c_nnz,
        c_csr_col_indices,
        c_csr_row_ptrs,
        c_csr_vals,
        c_lhss,
        c_rhss,
        c_lbs,
        c_ubs,
        c_vartypes,
        c_fullAsync
    )

    return list(c_lbs), list(c_ubs)


def propagateGPUAtomicWithMeasure(
        n_vars: int,
        n_cons: int,
        nnz: int,
        csr_col_indices: List[int],
        csr_row_ptrs: List[int],
        csr_vals: List[float],
        lhss: List[float],
        rhss: List[float],
        lbs: List[float],
        ubs: List[float],
        vartypes: List[int]
) -> Tuple[List[float]]:
    C = CDLL(so_file)

    check_input(n_vars, n_cons, nnz, csr_col_indices, csr_row_ptrs, csr_vals, lhss, rhss,
                lbs, ubs, vartypes)

    c_n_vars = c_int(n_vars)
    c_n_cons = c_int(n_cons)
    c_nnz = c_int(nnz)
    c_csr_col_indices = (c_int * nnz)(*csr_col_indices)
    c_csr_row_ptrs = (c_int * (n_cons + 1))(*csr_row_ptrs)
    c_csr_vals = (c_double * nnz)(*csr_vals)
    c_lhss = (c_double * n_cons)(*lhss)
    c_rhss = (c_double * n_cons)(*rhss)
    c_lbs = (c_double * n_vars)(*lbs)
    c_ubs = (c_double * n_vars)(*ubs)
    c_vartypes = (c_int * n_vars)(*vartypes)

    C.atomicPropagateWithMeasureDouble(
        c_n_cons,
        c_n_vars,
        c_nnz,
        c_csr_col_indices,
        c_csr_row_ptrs,
        c_csr_vals,
        c_lhss,
        c_rhss,
        c_lbs,
        c_ubs,
        c_vartypes
    )

    return list(c_lbs), list(c_ubs)


def propagateSequential(
        n_vars: int,
        n_cons: int,
        nnz: int,
        csr_col_indices: List[int],
        csr_row_ptrs: List[int],
        csr_vals: List[float],
        lhss: List[float],
        rhss: List[float],
        lbs: List[float],
        ubs: List[float],
        vartypes: List[int],
        datatype = c_double
) -> Tuple[List[float]]:
    C = CDLL(so_file)

    check_input(n_vars, n_cons, nnz, csr_col_indices, csr_row_ptrs, csr_vals, lhss, rhss,
                lbs, ubs, vartypes)

    c_n_vars = c_int(n_vars)
    c_n_cons = c_int(n_cons)
    c_nnz = c_int(nnz)
    c_csr_col_indices = (c_int * nnz)(*csr_col_indices)
    c_csr_row_ptrs = (c_int * (n_cons + 1))(*csr_row_ptrs)
    c_csr_vals = (datatype * nnz)(*csr_vals)
    c_lhss = (datatype * n_cons)(*lhss)
    c_rhss = (datatype * n_cons)(*rhss)
    c_lbs = (datatype * n_vars)(*lbs)
    c_ubs = (datatype * n_vars)(*ubs)
    c_vartypes = (c_int * n_vars)(*vartypes)

    if datatype == c_double:
        fun = C.propagateConstraintsSequentialDouble
    elif datatype == c_float:
        fun = C.propagateConstraintsSequentialFloat
    else:
        raise Exception("unsupported datatype")

    fun(
        c_n_cons,
        c_n_vars,
        c_nnz,
        c_csr_col_indices,
        c_csr_row_ptrs,
        c_csr_vals,
        c_lhss,
        c_rhss,
        c_lbs,
        c_ubs,
        c_vartypes
    )

    return list(c_lbs), list(c_ubs)


def propagateSequentialWithMeasure(
        n_vars: int,
        n_cons: int,
        nnz: int,
        csr_col_indices: List[int],
        csr_row_ptrs: List[int],
        csr_vals: List[float],
        lhss: List[float],
        rhss: List[float],
        lbs: List[float],
        ubs: List[float],
        vartypes: List[int]
) -> Tuple[List[float]]:
    C = CDLL(so_file)

    check_input(n_vars, n_cons, nnz, csr_col_indices, csr_row_ptrs, csr_vals, lhss, rhss,
                lbs, ubs, vartypes)

    c_n_vars = c_int(n_vars)
    c_n_cons = c_int(n_cons)
    c_nnz = c_int(nnz)
    c_csr_col_indices = (c_int * nnz)(*csr_col_indices)
    c_csr_row_ptrs = (c_int * (n_cons + 1))(*csr_row_ptrs)
    c_csr_vals = (c_double * nnz)(*csr_vals)
    c_lhss = (c_double * n_cons)(*lhss)
    c_rhss = (c_double * n_cons)(*rhss)
    c_lbs = (c_double * n_vars)(*lbs)
    c_ubs = (c_double * n_vars)(*ubs)
    c_vartypes = (c_int * n_vars)(*vartypes)

    C.sequentialPropagateWithMeasureDouble(
        c_n_cons,
        c_n_vars,
        c_nnz,
        c_csr_col_indices,
        c_csr_row_ptrs,
        c_csr_vals,
        c_lhss,
        c_rhss,
        c_lbs,
        c_ubs,
        c_vartypes
    )

    return list(c_lbs), list(c_ubs)


def propagateFullOMP(
        n_vars: int,
        n_cons: int,
        nnz: int,
        csr_col_indices: List[int],
        csr_row_ptrs: List[int],
        csr_vals: List[float],
        lhss: List[float],
        rhss: List[float],
        lbs: List[float],
        ubs: List[float],
        vartypes: List[int],
        datatype: _SimpleCData = c_double
) -> Tuple[List[float]]:
    C = CDLL(so_file)

    check_input(n_vars, n_cons, nnz, csr_col_indices, csr_row_ptrs, csr_vals, lhss, rhss,
                lbs, ubs, vartypes)

    c_n_vars = c_int(n_vars)
    c_n_cons = c_int(n_cons)
    c_nnz = c_int(nnz)
    c_csr_col_indices = (c_int * nnz)(*csr_col_indices)
    c_csr_row_ptrs = (c_int * (n_cons + 1))(*csr_row_ptrs)
    c_csr_vals = (datatype * nnz)(*csr_vals)
    c_lhss = (datatype * n_cons)(*lhss)
    c_rhss = (datatype * n_cons)(*rhss)
    c_lbs = (datatype * n_vars)(*lbs)
    c_ubs = (datatype * n_vars)(*ubs)
    c_vartypes = (c_int * n_vars)(*vartypes)

    if datatype == c_double:
        fun = C.propagateConstraintsFullOMPDouble
    elif datatype == c_float:
        fun = C.propagateConstraintsFullOMPFloat
    else:
        raise Exception("unsupported datatype")

    fun(
        c_n_cons,
        c_n_vars,
        c_nnz,
        c_csr_col_indices,
        c_csr_row_ptrs,
        c_csr_vals,
        c_lhss,
        c_rhss,
        c_lbs,
        c_ubs,
        c_vartypes
    )

    return list(c_lbs), list(c_ubs)


def propagateSequentialDisjoint(
        n_vars: int,
        n_cons: int,
        nnz: int,
        csr_col_indices: List[int],
        csr_row_ptrs: List[int],
        csr_vals: List[float],
        lhss: List[float],
        rhss: List[float],
        lbs: List[float],
        ubs: List[float],
        vartypes: List[int]
) -> Tuple[List[float]]:
    C = CDLL(so_file)

    check_input(n_vars, n_cons, nnz, csr_col_indices, csr_row_ptrs, csr_vals, lhss, rhss,
                lbs, ubs, vartypes)

    c_n_vars = c_int(n_vars)
    c_n_cons = c_int(n_cons)
    c_nnz = c_int(nnz)
    c_csr_col_indices = (c_int * nnz)(*csr_col_indices)
    c_csr_row_ptrs = (c_int * (n_cons + 1))(*csr_row_ptrs)
    c_csr_vals = (c_double * nnz)(*csr_vals)
    c_lhss = (c_double * n_cons)(*lhss)
    c_rhss = (c_double * n_cons)(*rhss)
    c_lbs = (c_double * n_vars)(*lbs)
    c_ubs = (c_double * n_vars)(*ubs)
    c_vartypes = (c_int * n_vars)(*vartypes)

    C.propagateConstraintsSequentialDisjointDouble(
        c_n_cons,
        c_n_vars,
        c_nnz,
        c_csr_col_indices,
        c_csr_row_ptrs,
        c_csr_vals,
        c_lhss,
        c_rhss,
        c_lbs,
        c_ubs,
        c_vartypes
    )

    return list(c_lbs), list(c_ubs)
