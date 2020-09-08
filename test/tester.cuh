#ifndef __GPUPROPAGATOR_PROPTESTER_CUH__
#define __GPUPROPAGATOR_PROPTESTER_CUH__

#include "test_infra.cuh"
#include "../src/propagators/GPU_propagator.cuh"
#include "../src/propagators/sequential_propagator.h"
#include "../src/propagators/OMP_propagator.h"
#include "test_setups.cuh"
#include "../src/GPU_interface.cuh"


template <typename datatype>
class Tester{
public:
    void executeSequentialPropagator(TestSetup<datatype>& ts)
    {
        GPUInterface gpu = GPUInterface();
        int* d_col_indices = gpu.initArrayGPU<int>   (ts.csr_col_indices, ts.nnz);
        int* d_row_ptrs    = gpu.initArrayGPU<int>   (ts.csr_row_ptrs,    ts.n_cons + 1);
        datatype* d_vals     = gpu.initArrayGPU<datatype>(ts.csr_vals,        ts.nnz);

        csr_to_csc(gpu, ts.n_cons, ts.n_vars, ts.nnz, d_col_indices, d_row_ptrs, ts.csc_col_ptrs, ts.csc_row_indices, ts.csc_vals, d_vals);

        sequentialPropagate<datatype>
        (
            ts.n_cons, ts.n_vars, ts.csr_col_indices, ts.csr_row_ptrs, ts.csc_col_ptrs, ts.csc_row_indices, ts.csr_vals,
            ts.lhss, ts.rhss, ts.lbs, ts.ubs, ts.vartypes
        );
    }

    void executeFullOMPPropagator(TestSetup<datatype>& ts)
    {
        GPUInterface gpu = GPUInterface();
        int* d_col_indices = gpu.initArrayGPU<int>   (ts.csr_col_indices, ts.nnz);
        int* d_row_ptrs    = gpu.initArrayGPU<int>   (ts.csr_row_ptrs,    ts.n_cons + 1);
        datatype* d_vals     = gpu.initArrayGPU<datatype>(ts.csr_vals,        ts.nnz);

        csr_to_csc(gpu, ts.n_cons, ts.n_vars, ts.nnz, d_col_indices, d_row_ptrs, ts.csc_col_ptrs, ts.csc_row_indices, ts.csc_vals, d_vals);

        fullOMPPropagate<datatype>
        (
            ts.n_cons, ts.n_vars, ts.csr_col_indices, ts.csr_row_ptrs, ts.csc_col_ptrs, ts.csc_row_indices, ts.csr_vals,
            ts.lhss, ts.rhss, ts.lbs, ts.ubs, ts.vartypes
        );
    }


    void executeSequentialDisjointPropagator(TestSetup<datatype>& ts)
    {
        GPUInterface gpu = GPUInterface();
        int* d_col_indices = gpu.initArrayGPU<int>   (ts.csr_col_indices, ts.nnz);
        int* d_row_ptrs    = gpu.initArrayGPU<int>   (ts.csr_row_ptrs,    ts.n_cons + 1);
        datatype* d_vals     = gpu.initArrayGPU<datatype>(ts.csr_vals,        ts.nnz);

        csr_to_csc(gpu, ts.n_cons, ts.n_vars, ts.nnz, d_col_indices, d_row_ptrs, ts.csc_col_ptrs, ts.csc_row_indices, ts.csc_vals, d_vals);

        sequentialPropagateDisjoint<datatype>
        (
            ts.n_cons, ts.n_vars, ts.csr_col_indices, ts.csr_row_ptrs, ts.csc_col_ptrs, ts.csc_row_indices, ts.csr_vals,
            ts.lhss, ts.rhss, ts.lbs, ts.ubs, ts.vartypes, ts.minacts, ts.maxacts, ts.maxactdeltas, ts.consmarked
        );
    }

    void executeFullGPUPropagator(TestSetup<datatype>& ts)
    {
        propagateConstraintsFullGPU<datatype>
        (
            ts.n_cons, ts.n_vars, ts.nnz, ts.csr_col_indices, ts.csr_row_ptrs, ts.csr_vals, ts.lhss, ts.rhss, ts.lbs, ts.ubs, ts.vartypes
        );
    }

    void executeAtomicGPUPropagator(TestSetup<datatype>& ts)
    {
        propagateConstraintsGPUAtomic<datatype>
        (
            ts.n_cons, ts.n_vars, ts.nnz, ts.csr_col_indices, ts.csr_row_ptrs, ts.csr_vals, ts.lhss, ts.rhss, ts.lbs, ts.ubs, ts.vartypes
        );
    }

    void checkSolution(TestSetup<datatype>& ts)
    {
        compareArrays<datatype>(ts.n_vars, ts.ubs,     ts.ubs_analytical);
        compareArrays<datatype>(ts.n_vars, ts.lbs,     ts.lbs_analytical);
    }
    
};

#endif