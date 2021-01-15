#ifndef __GPUPROPAGATOR_PROPTESTER_CUH__
#define __GPUPROPAGATOR_PROPTESTER_CUH__

#include "test_infra.cuh"
#include "../src/propagators/GPU_propagator.cuh"
#include "../src/propagators/sequential_propagator.h"
#include "../src/propagators/OMP_propagator.h"
#include "test_setups.cuh"
#include "../src/GPU_interface.cuh"


template<typename datatype>
class Tester {
public:
    void executeSequentialPropagator(TestSetup<datatype> &ts) {
       sequentialPropagate<datatype>
               (
                       ts.n_cons, ts.n_vars, ts.nnz, ts.csr_col_indices, ts.csr_row_ptrs, ts.csr_vals,
                       ts.lhss, ts.rhss, ts.lbs, ts.ubs, ts.vartypes
               );
    }

    void executeFullOMPPropagator(TestSetup<datatype> &ts) {
       fullOMPPropagate<datatype>
               (
                       ts.n_cons, ts.n_vars, ts.nnz, ts.csr_col_indices, ts.csr_row_ptrs, ts.csr_vals,
                       ts.lhss, ts.rhss, ts.lbs, ts.ubs, ts.vartypes
               );
    }


    void executeSequentialDisjointPropagator(TestSetup<datatype> &ts) {
       csr_to_csc(ts.n_cons, ts.n_vars, ts.nnz, ts.csr_col_indices, ts.csr_row_ptrs, ts.csc_col_ptrs, ts.csc_row_indices,
                  ts.csc_vals, ts.csr_vals);

       sequentialPropagateDisjoint<datatype>
               (
                       ts.n_cons, ts.n_vars, ts.csr_col_indices, ts.csr_row_ptrs, ts.csc_col_ptrs, ts.csc_row_indices,
                       ts.csr_vals,
                       ts.lhss, ts.rhss, ts.lbs, ts.ubs, ts.vartypes, ts.minacts, ts.maxacts, ts.maxactdeltas,
                       ts.consmarked
               );
    }

    void executeGPUReduction(TestSetup<datatype> &ts) {
       propagateConstraintsGPUReduction<datatype>
               (
                       ts.n_cons, ts.n_vars, ts.nnz, ts.csr_col_indices, ts.csr_row_ptrs, ts.csr_vals, ts.lhss, ts.rhss,
                       ts.lbs, ts.ubs, ts.vartypes
               );
    }

    void executeAtomicGPUPropagator(TestSetup<datatype> &ts) {
       propagateConstraintsGPUAtomic<datatype>
               (
                       ts.n_cons, ts.n_vars, ts.nnz, ts.csr_col_indices, ts.csr_row_ptrs, ts.csr_vals, ts.lhss, ts.rhss,
                       ts.lbs, ts.ubs, ts.vartypes
               );
    }

    void checkSolution(TestSetup<datatype> &ts) {
       compareArrays<datatype>(ts.n_vars, ts.ubs, ts.ubs_analytical, TEST_EPS);
       compareArrays<datatype>(ts.n_vars, ts.lbs, ts.lbs_analytical, TEST_EPS);
    }
};

#endif