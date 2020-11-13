from readerInterface import FileReaderInterface, get_reader
from GPUDomPropInterface import propagateGPU, propagateGPUAtomic, propagateSequential, propagateSequentialDisjoint, propagateFullOMP
from utils import print_bounds, compare_arrays_diff_idx, num_inf_bounds
import numpy as np
from typing import List
import argparse
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt


def vis_sparsity_pattern(m: int, n: int, col_indices: List[float], row_ptrs: List[float], coeffs: List[float]) -> None:
    A = csr_matrix((coeffs, col_indices, row_ptrs), shape=(m, n)).toarray()
    plt.spy(A, markersize=1)
    plt.show()
    exit(1)


def normalize_infs(arr: List[float]) -> List[float]:
    arr = map(lambda val: 1e20  if val >= 1e20  else val, arr)
    arr = map(lambda val: -1e20 if val <= -1e20 else val, arr)
    return list(arr)


def arrays_equal(bds1: List[float], bds2: List[float]) -> bool:
    return np.isclose(bds1, bds2).all()


def prop_compare_seq_gpu(lp_file_path: str) -> None:
    reader: FileReaderInterface = get_reader(lp_file_path)

    n_cons = reader.get_n_cons()
    n_vars = reader.get_n_vars()
    nnz = reader.get_nnz()
    lbs, ubs = reader.get_var_bounds()
    lhss, rhss = reader.get_lrhss()
    coeffs, row_ptrs, col_indices = reader.get_cons_matrix()
    vartypes = reader.get_SCIP_vartypes()

    # print sparsity and input data size
    print("num vars: ", n_vars)
    print("num cons: ", n_cons)
    print("nnz     : ", nnz)
    print("Num inf bounds before propagation: ", num_inf_bounds(lbs, ubs))

    lbs_seq = lbs_gpuatomic = lbs_gpu = lbs_omp = lbs
    ubs_seq = ubs_gpuatomic = ubs_gpu = ubs_omp = ubs

   # (seq_new_lbs, seq_new_ubs) = propagateSequential( n_vars, n_cons, nnz, col_indices, row_ptrs, coeffs, lhss, rhss, lbs_seq, ubs_seq, vartypes)

   # (omp_new_lbs, omp_new_ubs) = propagateFullOMP( n_vars, n_cons, nnz, col_indices, row_ptrs, coeffs, lhss, rhss, lbs_omp, ubs_omp, vartypes)

    (gpu_new_lbs, gpu_new_ubs) = propagateGPU( n_vars, n_cons, nnz, col_indices, row_ptrs, coeffs, lhss, rhss, lbs_gpu, ubs_gpu, vartypes)

    (gpuatomic_new_lbs, gpuatomic_new_ubs) = propagateGPUAtomic( n_vars, n_cons, nnz, col_indices, row_ptrs, coeffs, lhss, rhss, lbs_gpuatomic, ubs_gpuatomic, vartypes)

    print("Num inf bounds after atomic propagation: ", num_inf_bounds(gpuatomic_new_lbs, gpuatomic_new_ubs))

    #seq_new_lbs = normalize_infs(seq_new_lbs)
    #seq_new_ubs = normalize_infs(seq_new_ubs)
    gpu_new_lbs = normalize_infs(gpu_new_lbs)
    gpu_new_ubs = normalize_infs(gpu_new_ubs)
    #omp_new_lbs = normalize_infs(omp_new_lbs)
    #omp_new_ubs = normalize_infs(omp_new_ubs)
    gpuatomic_new_lbs = normalize_infs(gpuatomic_new_lbs)
    gpuatomic_new_ubs = normalize_infs(gpuatomic_new_ubs)

    #equal_seq_gpu_atomic = arrays_equal(seq_new_lbs, gpuatomic_new_lbs) and arrays_equal(seq_new_ubs, gpuatomic_new_ubs)
    #equal_seq_gpu_full = arrays_equal(seq_new_lbs, gpu_new_lbs) and arrays_equal(seq_new_ubs, gpu_new_ubs)
    #equal_seq_omp = arrays_equal(seq_new_lbs, omp_new_lbs) and arrays_equal(seq_new_ubs, omp_new_ubs)

   # print("\ncpu_seq to cpu_omp results match: ", equal_seq_omp)
   # print("cpu_seq to gpu_reduction results match: ", equal_seq_gpu_full)
   # print("cpu_seq to gpu_atomic results match: ", equal_seq_gpu_atomic)
    #print("all results match: ", equal_seq_gpu_atomic and equal_seq_gpu_full and equal_seq_omp)

    print("gpu_reduction to gpu_atomic results match: ", arrays_equal(gpu_new_lbs, gpuatomic_new_lbs) and arrays_equal(gpu_new_ubs, gpuatomic_new_ubs))
  #  compare_arrays_diff_idx(gpu_new_lbs, gpuatomic_new_lbs, "lbs")
  #  compare_arrays_diff_idx(gpu_new_ubs, gpuatomic_new_ubs, "ubs")
    #print_bounds(seq_new_lbs, seq_new_ubs)
    #print_bounds(gpuatomic_new_lbs, gpuatomic_new_ubs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Propagate MIP or LP file')
    parser.add_argument("-f", "--file", type=str, required=True)
    args = parser.parse_args()

    try:
        prop_compare_seq_gpu(args.file)
    except Exception as e:
        print("\nexecution of ", args.file, " failed. Exception: ")
        print(e)
