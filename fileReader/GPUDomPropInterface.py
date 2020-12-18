from ctypes import *
from typing import List, Tuple

so_file = "../build/libGpuProp.so"


def propagateGPU(
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

    assert len(csr_col_indices) == nnz
    assert len(csr_row_ptrs) == n_cons + 1
    assert len(csr_vals) == nnz
    assert len(lhss) == n_cons
    assert len(rhss) == n_cons
    assert len(lbs) == n_vars
    assert len(ubs) == n_vars
    assert len(vartypes) == n_vars

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

    C.propagateConstraintsFullGPUdouble(
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
        vartypes: List[int]
) -> Tuple[List[float]]:
    C = CDLL(so_file)

    assert len(csr_col_indices) == nnz
    assert len(csr_row_ptrs) == n_cons + 1
    assert len(csr_vals) == nnz
    assert len(lhss) == n_cons
    assert len(rhss) == n_cons
    assert len(lbs) == n_vars
    assert len(ubs) == n_vars
    assert len(vartypes) == n_vars

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

    C.propagateConstraintsGPUAtomicDouble(
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

    assert len(csr_col_indices) == nnz
    assert len(csr_row_ptrs) == n_cons + 1
    assert len(csr_vals) == nnz
    assert len(lhss) == n_cons
    assert len(rhss) == n_cons
    assert len(lbs) == n_vars
    assert len(ubs) == n_vars
    assert len(vartypes) == n_vars

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
        vartypes: List[int]
) -> Tuple[List[float]]:
    C = CDLL(so_file)

    assert len(csr_col_indices) == nnz
    assert len(csr_row_ptrs) == n_cons + 1
    assert len(csr_vals) == nnz
    assert len(lhss) == n_cons
    assert len(rhss) == n_cons
    assert len(lbs) == n_vars
    assert len(ubs) == n_vars
    assert len(vartypes) == n_vars

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

    C.propagateConstraintsSequentialDouble(
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

    assert len(csr_col_indices) == nnz
    assert len(csr_row_ptrs) == n_cons + 1
    assert len(csr_vals) == nnz
    assert len(lhss) == n_cons
    assert len(rhss) == n_cons
    assert len(lbs) == n_vars
    assert len(ubs) == n_vars
    assert len(vartypes) == n_vars

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
        vartypes: List[int]
) -> Tuple[List[float]]:
    C = CDLL(so_file)

    assert len(csr_col_indices) == nnz
    assert len(csr_row_ptrs) == n_cons + 1
    assert len(csr_vals) == nnz
    assert len(lhss) == n_cons
    assert len(rhss) == n_cons
    assert len(lbs) == n_vars
    assert len(ubs) == n_vars
    assert len(vartypes) == n_vars

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

    C.propagateConstraintsFullOMPDouble(
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

    assert len(csr_col_indices) == nnz
    assert len(csr_row_ptrs) == n_cons + 1
    assert len(csr_vals) == nnz
    assert len(lhss) == n_cons
    assert len(rhss) == n_cons
    assert len(lbs) == n_vars
    assert len(ubs) == n_vars
    assert len(vartypes) == n_vars

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
