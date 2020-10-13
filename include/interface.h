#ifndef __GPUPROPAGATOR_INTERFACE_CUH__
#define __GPUPROPAGATOR_INTERFACE_CUH__


#ifdef __cplusplus
extern "C" {
#endif

void propagateConstraintsFullGPUdouble(
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

void propagateConstraintsGPUAtomicDouble(
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

void propagateConstraintsSequentialDouble
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

void propagateConstraintsFullOMPDouble
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

void propagateConstraintsSequentialDisjointDouble
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
