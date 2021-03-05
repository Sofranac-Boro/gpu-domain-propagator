#ifndef __GPUPROPAGATOR_INTERFACE_CUH__
#define __GPUPROPAGATOR_INTERFACE_CUH__

#include "../src/def.h"

#ifdef __cplusplus
extern "C" {
#endif

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
);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

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
);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

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
        const bool fullAsync = true
);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

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
);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

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
        );

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

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
        );

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

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
        );

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

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
        );

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

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
        );

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

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
        );

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

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
        );

#ifdef __cplusplus
}
#endif

#endif
