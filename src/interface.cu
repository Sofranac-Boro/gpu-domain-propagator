#include "../include/interface.h"
#include "propagators/GPU_propagator.cuh"
#include "propagators/sequential_propagator.h"
#include "propagators/OMP_propagator.h"
#include "progressMeasureMethods/progressPropagators.cuh"

#define CALL_AND_HANDLE(expr)                     \
    try {                                         \
        return (expr);                            \
    }                                             \
   catch (const std::exception &exc)              \
   {                                              \
      std::cerr <<"Error happened. Exception:\n"; \
      std::cerr << exc.what() << "\n";            \
      return GDP_ERROR;                           \
   }

GDP_RETCODE propagateConstraintsGPUReductionDouble(
        const int n_cons,
        const int n_vars,
        const int nnz,
        const int *csr_col_indices,
        const int *csr_row_ptrs,
        const double *csr_vals,
        const double *lhss,
        const double *rhss,
        double *lbs,
        double *ubs,
        const int *vartypes
) {
      CALL_AND_HANDLE(
              propagateConstraintsGPUReduction<double>
                      (
                              n_cons,
                              n_vars,
                              nnz,
                              csr_col_indices,
                              csr_row_ptrs,
                              csr_vals,
                              lhss,
                              rhss,
                              lbs,
                              ubs,
                              (GDP_VARTYPE *) vartypes
                      )
              )
}

GDP_RETCODE propagateConstraintsGPUReductionFloat(
        const int n_cons,
        const int n_vars,
        const int nnz,
        const int *csr_col_indices,
        const int *csr_row_ptrs,
        const float *csr_vals,
        const float *lhss,
        const float *rhss,
        float *lbs,
        float *ubs,
        const int *vartypes
) {
   CALL_AND_HANDLE(
           propagateConstraintsGPUReduction<float>
                   (
                           n_cons,
                           n_vars,
                           nnz,
                           csr_col_indices,
                           csr_row_ptrs,
                           csr_vals,
                           lhss,
                           rhss,
                           lbs,
                           ubs,
                           (GDP_VARTYPE *) vartypes
                   )
   )
}

GDP_RETCODE propagateConstraintsGPUAtomicDouble(
        const int n_cons,
        const int n_vars,
        const int nnz,
        const int *csr_col_indices,
        const int *csr_row_ptrs,
        const double *csr_vals,
        const double *lhss,
        const double *rhss,
        double *lbs,
        double *ubs,
        const int *vartypes,
        const bool fullAsync
) {
      CALL_AND_HANDLE( propagateConstraintsGPUAtomic<double>
              (
                      n_cons,
                      n_vars,
                      nnz,
                      csr_col_indices,
                      csr_row_ptrs,
                      csr_vals,
                      lhss,
                      rhss,
                      lbs,
                      ubs,
                      (GDP_VARTYPE *) vartypes,
                      fullAsync
              ))
}

GDP_RETCODE propagateConstraintsGPUAtomicFloat(
        const int n_cons,
        const int n_vars,
        const int nnz,
        const int *csr_col_indices,
        const int *csr_row_ptrs,
        const float *csr_vals,
        const float *lhss,
        const float *rhss,
        float *lbs,
        float *ubs,
        const int *vartypes,
        const bool fullAsync
) {
      CALL_AND_HANDLE( propagateConstraintsGPUAtomic<float>
              (
                      n_cons,
                      n_vars,
                      nnz,
                      csr_col_indices,
                      csr_row_ptrs,
                      csr_vals,
                      lhss,
                      rhss,
                      lbs,
                      ubs,
                      (GDP_VARTYPE *) vartypes,
                      fullAsync
              ) )
}

GDP_RETCODE propagateConstraintsSequentialDouble
        (
                const int n_cons,
                const int n_vars,
                const int nnz,
                const int *col_indices,
                const int *row_indices,
                const double *vals,
                const double *lhss,
                const double *rhss,
                double *lbs,
                double *ubs,
                const int *vartypes
        ) {
      CALL_AND_HANDLE(sequentialPropagate<double>
              (
                      n_cons,
                      n_vars,
                      nnz,
                      col_indices,
                      row_indices,
                      vals,
                      lhss,
                      rhss,
                      lbs,
                      ubs,
                      (GDP_VARTYPE *) vartypes
              ))
}

GDP_RETCODE propagateConstraintsSequentialFloat
        (
                const int n_cons,
                const int n_vars,
                const int nnz,
                const int *col_indices,
                const int *row_indices,
                const float *vals,
                const float *lhss,
                const float *rhss,
                float *lbs,
                float *ubs,
                const int *vartypes
        ) {
      CALL_AND_HANDLE(sequentialPropagate<float>
              (
                      n_cons,
                      n_vars,
                      nnz,
                      col_indices,
                      row_indices,
                      vals,
                      lhss,
                      rhss,
                      lbs,
                      ubs,
                      (GDP_VARTYPE *) vartypes
              ))
}

GDP_RETCODE propagateConstraintsFullOMPDouble
        (
                const int n_cons,
                const int n_vars,
                const int nnz,
                const int *col_indices,
                const int *row_indices,
                const double *vals,
                const double *lhss,
                const double *rhss,
                double *lbs,
                double *ubs,
                const int *vartypes
        ) {
      CALL_AND_HANDLE(fullOMPPropagate<double>
              (
                      n_cons,
                      n_vars,
                      nnz,
                      col_indices,
                      row_indices,
                      vals,
                      lhss,
                      rhss,
                      lbs,
                      ubs,
                      (GDP_VARTYPE *) vartypes
              ))
}

GDP_RETCODE propagateConstraintsFullOMPFloat
        (
                const int n_cons,
                const int n_vars,
                const int nnz,
                const int *col_indices,
                const int *row_indices,
                const float *vals,
                const float *lhss,
                const float *rhss,
                float *lbs,
                float *ubs,
                const int *vartypes
        ) {
   CALL_AND_HANDLE(fullOMPPropagate<float>
                           (
                                   n_cons,
                                   n_vars,
                                   nnz,
                                   col_indices,
                                   row_indices,
                                   vals,
                                   lhss,
                                   rhss,
                                   lbs,
                                   ubs,
                                   (GDP_VARTYPE *) vartypes
                           ))
}

GDP_RETCODE propagateConstraintsSequentialDisjointDouble
        (
                const int n_cons,
                const int n_vars,
                const int nnz,
                const int *col_indices,
                const int *row_indices,
                const double *vals,
                const double *lhss,
                const double *rhss,
                double *lbs,
                double *ubs,
                const int *vartypes
        ) {

      CALL_AND_HANDLE(sequentialPropagateDisjoint<double>
              (
                      n_cons,
                      n_vars,
                      nnz,
                      col_indices,
                      row_indices,
                      vals,
                      lhss,
                      rhss,
                      lbs,
                      ubs,
                      (GDP_VARTYPE *) vartypes
              ))
}

GDP_RETCODE sequentialPropagateWithMeasureDouble
        (
                const int n_cons,
                const int n_vars,
                const int nnz,
                const int *col_indices,
                const int *row_indices,
                const double *vals,
                const double *lhss,
                const double *rhss,
                double *lbs,
                double *ubs,
                const int *vartypes
        ) {

      CALL_AND_HANDLE(sequentialPropagateWithMeasure<double>
              (
                      n_cons,
                      n_vars,
                      nnz,
                      col_indices,
                      row_indices,
                      vals,
                      lhss,
                      rhss,
                      lbs,
                      ubs,
                      (GDP_VARTYPE *) vartypes
              ))
}

GDP_RETCODE atomicPropagateWithMeasureDouble
        (
                const int n_cons,
                const int n_vars,
                const int nnz,
                const int *col_indices,
                const int *row_indices,
                const double *vals,
                const double *lhss,
                const double *rhss,
                double *lbs,
                double *ubs,
                const int *vartypes
        ) {

      CALL_AND_HANDLE(propagateConstraintsGPUAtomicWithMeasure<double>
              (
                      n_cons,
                      n_vars,
                      nnz,
                      col_indices,
                      row_indices,
                      vals,
                      lhss,
                      rhss,
                      lbs,
                      ubs,
                      (GDP_VARTYPE *) vartypes
              ))
}

