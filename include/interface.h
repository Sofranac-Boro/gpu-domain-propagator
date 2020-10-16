#ifndef __GPUPROPAGATOR_INTERFACE_CUH__
#define __GPUPROPAGATOR_INTERFACE_CUH__

#include "../src/def.h"

#ifdef __cplusplus
extern "C" {
#endif

GDP_RETCODE propagateConstraintsFullGPUdouble(
        const int n_cons,
        const int n_vars,
        const int nnz,
        int *csr_col_indices,
        int *csr_row_ptrs,
        double *csr_vals,
        double *lhss,
        double *rhss,
        double *lbs,
        double *ubs,
        int *vartypes
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
        int *csr_col_indices,
        int *csr_row_ptrs,
        double *csr_vals,
        double *lhss,
        double *rhss,
        double *lbs,
        double *ubs,
        int *vartypes
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

#endif
