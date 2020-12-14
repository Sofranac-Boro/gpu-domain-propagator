#ifndef GPU_DOMAIN_PROPAGATOR_PREPROCESSOR_H
#define GPU_DOMAIN_PROPAGATOR_PREPROCESSOR_H

#include "sequential_propagator.h"

template<class datatype>
void preprocessExecute
        (
                const int n_cons,
                const int n_vars,
                const int *col_indices,
                const int *row_indices,
                const int *csc_col_ptrs,
                const int *csc_row_indices,
                const datatype *vals,
                const datatype *lhss,
                const datatype *rhss,
                datatype *lbs,
                datatype *ubs,
                const GDP_VARTYPE *vartypes
        ) {

   DEBUG_CALL(checkInput(n_cons, n_vars, row_indices[n_cons], vals, lhss, rhss, lbs, ubs, vartypes));

   datatype *minacts = (datatype *) calloc(n_cons, sizeof(datatype));
   datatype *maxacts = (datatype *) calloc(n_cons, sizeof(datatype));
   datatype *maxactdeltas = (datatype *) calloc(n_cons, sizeof(datatype));
   int *consmarked = (int *) calloc(n_cons, sizeof(int));

#ifdef VERBOSE
   auto start = std::chrono::steady_clock::now();
#endif

   VERBOSE_CALL(printf("\npreprocessor execution start... Params: MAXNUMROUNDS: %d", MAX_NUM_ROUNDS));

   bool change_found = true;
   int prop_round = 0;
   for (prop_round = 1; prop_round <= MAX_NUM_ROUNDS && change_found; prop_round++)  // maxnumrounds = 100
   {
      VERBOSE_CALL_2(printf("\nPropagation round: %d\n\n", prop_round));

      sequentialComputeActivities<datatype>(n_cons, col_indices, row_indices, vals, ubs, lbs, minacts, maxacts,
                                            maxactdeltas);

      change_found = sequentialPropagationRound<datatype>
              (
                      n_cons, n_vars, col_indices, row_indices, csc_col_ptrs, csc_row_indices, vals, lhss, rhss,
                      lbs, ubs, vartypes, minacts, maxacts, maxactdeltas, consmarked, RECOMPUTE_ACTS_FALSE
              );

   }

   VERBOSE_CALL(printf("\ncpu_seq_dis propagation done. Num rounds: %d\n", prop_round - 1));
   VERBOSE_CALL(measureTime("cpu_seq_dis", start, std::chrono::steady_clock::now()));

   free(minacts);
   free(maxacts);
   free(maxactdeltas);
   free(consmarked);
}

#endif //GPU_DOMAIN_PROPAGATOR_PREPROCESSOR_H
