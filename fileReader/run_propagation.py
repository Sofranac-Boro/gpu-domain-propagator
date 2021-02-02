import argparse
from ctypes import c_float, c_double, _SimpleCData

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from typing import List


from papiloInterface import PapiloInterface
from GPUDomPropInterface import propagateGPUReduction, propagateGPUAtomic, propagateSequential, propagateFullOMP, \
propagateSequentialWithMeasure, propagateGPUAtomicWithMeasure, propagateSequentialDisjoint
from readerInterface import FileReaderInterface, get_reader
from regexes import OutputGrabber
from utils import plot_progress_save_pdf, compare_arrays_diff_idx


def vis_sparsity_pattern(m: int, n: int, col_indices: List[float], row_ptrs: List[float], coeffs: List[float]) -> None:
    A = csr_matrix((coeffs, col_indices, row_ptrs), shape=(m, n)).toarray()
    plt.spy(A, markersize=1)
    plt.show()
    exit(1)


def normalize_infs(arr: List[float]) -> List[float]:
    arr = list(map(lambda val: 1e20 if val >= 1e20 else val, arr))
    arr = list(map(lambda val: -1e20 if val <= -1e20 else val, arr))
    return arr


def arrays_equal(bds1: List[float], bds2: List[float]) -> bool:
    return np.isclose(bds1, bds2).all()


def prop_compare_seq_gpu(lp_file_path: str, datatype: _SimpleCData = c_double) -> None:
    reader: FileReaderInterface = get_reader(lp_file_path)

    n_cons = reader.get_n_cons()
    n_vars = reader.get_n_vars()
    nnz = reader.get_nnz()
    lbs, ubs = reader.get_var_bounds()
    lhss, rhss = reader.get_lrhss()
    coeffs, row_ptrs, col_indices = reader.get_cons_matrix()
    vartypes = reader.get_SCIP_vartypes()

    print("lbs: ", lbs)
    print("ubs: ", ubs)

   # print("input lbs: ", ubs)

    # print sparsity and input data size
    print("num vars: ", n_vars)
    print("num cons: ", n_cons)
    print("nnz     : ", nnz)

    lbs_dis = lbs_seq = lbs_gpuatomic = lbs_gpu = lbs_omp = lbs
    ubs_dis = ubs_seq = ubs_gpuatomic = ubs_gpu = ubs_omp = ubs

    (seq_new_lbs, seq_new_ubs) = propagateSequential(n_vars, n_cons, nnz, col_indices, row_ptrs, coeffs, lhss, rhss,
                                                     lbs_seq, ubs_seq, vartypes, datatype=c_double)

 #   (omp_new_lbs, omp_new_ubs) = propagateFullOMP(n_vars, n_cons, nnz, col_indices, row_ptrs, coeffs, lhss, rhss, lbs_omp, ubs_omp, vartypes, datatype=c_double)

 #   (gpu_new_lbs, gpu_new_ubs) = propagateGPUReduction(n_vars, n_cons, nnz, col_indices, row_ptrs, coeffs, lhss, rhss, lbs_gpu, ubs_gpu, vartypes, datatype=datatype)

   # (dis_new_lbs, dis_new_ubs) = propagateSequentialDisjoint( n_vars, n_cons, nnz, col_indices, row_ptrs, coeffs, lhss, rhss, lbs_dis, ubs_dis, vartypes)

   # (gpuatomic_new_lbs, gpuatomic_new_ubs) = propagateGPUAtomic(n_vars, n_cons, nnz, col_indices, row_ptrs, coeffs,
  #                                                              lhss, rhss, lbs_gpuatomic, ubs_gpuatomic, vartypes, fullAsync=False, datatype=datatype)

    seq_new_lbs = normalize_infs(seq_new_lbs)
    seq_new_ubs = normalize_infs(seq_new_ubs)
    # gpu_new_lbs = normalize_infs(gpu_new_lbs)
    # gpu_new_ubs = normalize_infs(gpu_new_ubs)
    # omp_new_lbs = normalize_infs(omp_new_lbs)
    # omp_new_ubs = normalize_infs(omp_new_ubs)
    # gpuatomic_new_lbs = normalize_infs(gpuatomic_new_lbs)
    # gpuatomic_new_ubs = normalize_infs(gpuatomic_new_ubs)
    # # dis_new_lbs = normalize_infs(dis_new_lbs)
    # # dis_new_ubs = normalize_infs(dis_new_ubs)
    #
    # equal_seq_gpu_atomic = arrays_equal(seq_new_lbs, gpuatomic_new_lbs) and arrays_equal(seq_new_ubs, gpuatomic_new_ubs)
    # equal_seq_gpu_full = arrays_equal(seq_new_lbs, gpu_new_lbs) and arrays_equal(seq_new_ubs, gpu_new_ubs)
    # equal_seq_omp = arrays_equal(seq_new_lbs, omp_new_lbs) and arrays_equal(seq_new_ubs, omp_new_ubs)
    # # equal_seq_dis = arrays_equal(seq_new_lbs, dis_new_lbs) and arrays_equal(seq_new_ubs, dis_new_ubs)
    # print("\ncpu_seq to cpu_omp results match: ", equal_seq_omp)
    # print("cpu_seq to gpu_reduction results match: ", equal_seq_gpu_full)
    # print("cpu_seq to gpu_atomic results match: ", equal_seq_gpu_atomic)
    # # print("cpu_seq to cpu_seq_dis results match: ", equal_seq_dis)
    # print("all results match: ", equal_seq_gpu_atomic and equal_seq_gpu_full and equal_seq_omp)

    papilo = PapiloInterface("/home/bzfsofra/papilo", lp_file_path)
    papilo.run_papilo()
    lbs_papilo, ubs_papilo = papilo.get_presolved_bounds()
    lbs_papilo = normalize_infs(lbs_papilo)
    ubs_papilo = normalize_infs(ubs_papilo)
    print("papilo rounds: ", papilo.num_rounds, " time (s): ", papilo.exec_time)
   # print("seq lbs: ", seq_new_lbs)
   # print("pap lbs: ", lbs_papilo)
   # print("seq ubs: ", seq_new_ubs)
   # print("pap ubs: ", ubs_papilo)
    #print(ubs)

    equal_seq_papilo = arrays_equal(seq_new_lbs, lbs_papilo) and arrays_equal(seq_new_ubs, ubs_papilo)
    print("cpu_seq to pailo results match: ", equal_seq_papilo)

   # print("cpu_seq lbs: ", seq_new_ubs)

    compare_arrays_diff_idx(seq_new_lbs, lbs_papilo, "lbs")
    compare_arrays_diff_idx(seq_new_ubs, ubs_papilo, "ubs")
# print_bounds(seq_new_lbs, seq_new_ubs)
# print_bounds(gpuatomic_new_lbs, gpuatomic_new_ubs)


def propagation_measure_run(input_file: str):
    out = OutputGrabber()

    with out:
        reader: FileReaderInterface = get_reader(input_file)

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

        lbs_seq = lbs_gpuatomic = lbs
        ubs_seq = ubs_gpuatomic = ubs

        (seq_new_lbs, seq_new_ubs) = propagateSequentialWithMeasure(n_vars, n_cons, nnz, col_indices, row_ptrs, coeffs, lhss, rhss,
                                                                    lbs_seq, ubs_seq, vartypes)

        (gpuatomic_new_lbs, gpuatomic_new_ubs) = propagateGPUAtomicWithMeasure(n_vars, n_cons, nnz, col_indices, row_ptrs, coeffs,
                                                                               lhss, rhss, lbs_gpuatomic, ubs_gpuatomic, vartypes)

        seq_new_lbs = normalize_infs(seq_new_lbs)
        seq_new_ubs = normalize_infs(seq_new_ubs)
        gpuatomic_new_lbs = normalize_infs(gpuatomic_new_lbs)
        gpuatomic_new_ubs = normalize_infs(gpuatomic_new_ubs)

        equal_seq_gpu_atomic = arrays_equal(seq_new_lbs, gpuatomic_new_lbs) and arrays_equal(seq_new_ubs, gpuatomic_new_ubs)
        print("cpu_seq to gpu_atomic results match: ", equal_seq_gpu_atomic)

    print(out.capturedtext)

    plot_progress_save_pdf(out.capturedtext)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Propagate MIP or LP file')
    parser.add_argument("-f", "--file", type=str, required=True)
    parser.add_argument("-d", "--datatype", type=str, required=False, default="double")
    args = parser.parse_args()

    if args.datatype == "" or args.datatype == "double":
        datatype = c_double
    elif args.datatype == "float":
        datatype = c_float
    else:
        raise Exception("Unsupported datatype: ", args.datatype)

    try:
        prop_compare_seq_gpu(args.file, datatype)
       # propagation_measure_run(args.file)
    except Exception as e:
        print("\nexecution of ", args.file, " failed. Exception: ")
        print(e)
