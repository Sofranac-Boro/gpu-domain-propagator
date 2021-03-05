#include "propagation_methods.cuh"

void markConstraints
        (
                const int var_idx,
                const int *csc_col_ptrs,
                const int *csc_row_indices,
                int *consmarked
        ) {
   for (int i = csc_col_ptrs[var_idx]; i < csc_col_ptrs[var_idx + 1]; i++) {
      consmarked[csc_row_indices[i]] = 1;
   }
}