#include "propagation_methods.h"
#include <stdio.h>

void markConstraints
        (
                const int var_idx,
                const int *csc_col_ptrs,
                const int *csc_row_indices,
                int *consmarked
        ) {
   //printf("var_idx: %d appears in cons:", var_idx);
   // cons_indices_begin = csc_col_ptrs[var_idx];
   // cons_indices_end   = csc_col_ptrs[var_idx+1];  
   for (int i = csc_col_ptrs[var_idx]; i < csc_col_ptrs[var_idx + 1]; i++) {
      // cons_idx = csc_row_indices[i];
      //  printf("%d ", csc_row_indices[i]);

      consmarked[csc_row_indices[i]] = 1;
   }
   //printf("\n");

}