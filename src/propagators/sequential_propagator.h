
#ifndef __GPUPROPAGATOR_SEQUENTIAL_CUH__
#define __GPUPROPAGATOR_SEQUENTIAL_CUH__

#include "../propagation_methods.cuh"
#include "../misc.h"
#include "../params.h"

#define RECOMPUTE_ACTS_TRUE true
#define RECOMPUTE_ACTS_FALSE false


template<class datatype>
void sequentialComputeActivities
        (
                const int n_cons,
                const int *col_indices,
                const int *row_indices,
                const datatype *vals,
                const datatype *ubs,
                const datatype *lbs,
                datatype *minacts,
                datatype *maxacts,
                datatype *maxactdeltas
        ) {
   for (int considx = 0; considx < n_cons; considx++) {
      ActivitiesTuple activities = computeActivities(considx, col_indices, row_indices, vals, ubs, lbs);
      minacts[considx] = activities.minact;
      maxacts[considx] = activities.maxact;
      maxactdeltas[considx] = activities.maxactdelta;
   }
}


template<class datatype>
bool sequentialPropagationRound
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
                const int *vartypes,
                datatype *minacts,
                datatype *maxacts,
                datatype *maxactdeltas,
                int *consmarked,
                const bool recomputeActs
        ) {

   datatype coeff;
   bool isVarCont;
   datatype rhs;
   datatype lhs;
   datatype slack;
   datatype surplus;

   bool change_found = false;
   int val_idx;
   int var_idx;

   int *consmarked_nextround = (int *) calloc(n_cons, sizeof(int));

   for (int considx = 0; considx < n_cons; considx++) {
      if (consmarked[considx] == 1) {
         if (recomputeActs) {
            ActivitiesTuple activities = computeActivities(considx, col_indices, row_indices, vals, ubs, lbs);
            minacts[considx] = activities.minact;
            maxacts[considx] = activities.maxact;
            maxactdeltas[considx] = activities.maxactdelta;
         }

         DEBUG_CALL( printf("cons %d: minact: %.5f, maxact: %.5f\n", considx, minacts[considx], maxacts[considx]) );

         rhs = rhss[considx];
         lhs = lhss[considx];
         slack = rhs - minacts[considx];
         surplus = maxacts[considx] - lhs;

         if (canConsBeTightened(slack, surplus, maxactdeltas[considx])) {
            int num_vars_in_cons = row_indices[considx + 1] - row_indices[considx];
            slack = slack < 0.0 ? 0.0 : slack;

            for (int var = 0; var < num_vars_in_cons; var++) {
               val_idx = row_indices[considx] + var;
               var_idx = col_indices[val_idx];
               coeff = vals[val_idx];

               isVarCont = vartypes[var_idx] == 3;

               bool tightened = tightenVariable<datatype>
                       (
                               coeff, lhs, rhs, minacts[considx], maxacts[considx], isVarCont, var_idx, val_idx,
                               csc_col_ptrs, csc_row_indices, consmarked_nextround, lbs, ubs
                       );

               change_found = tightened ? tightened : change_found;
            }
         }
      }
   }

   // copy data from consmarked_nextround into consmarked
   memcpy(consmarked, consmarked_nextround, n_cons * sizeof(int));
   free(consmarked_nextround);
   return change_found;
}

template<class datatype>
void sequentialPropagateDisjoint
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
                const int *vartypes
        ) {

   datatype *minacts = (datatype *) calloc(n_cons, sizeof(datatype));
   datatype *maxacts = (datatype *) calloc(n_cons, sizeof(datatype));
   datatype *maxactdeltas = (datatype *) calloc(n_cons, sizeof(datatype));
   int *consmarked = (int *) calloc(n_cons, sizeof(int));

#ifdef VERBOSE
   auto start = std::chrono::steady_clock::now();
#endif

   VERBOSE_CALL(printf("\nStarting cpu_seq exectution!\n"));

   bool change_found = true;
   int prop_round = 0;
   for (prop_round = 1; prop_round <= MAX_NUM_ROUNDS && change_found; prop_round++)  // maxnumrounds = 100
   {
      DEBUG_CALL(printf("propagation round: %d\n", prop_round));
      //VERBOSE_CALL( countPrintNumMarkedCons<int>(n_cons, consmarked) );

      sequentialComputeActivities<datatype>(n_cons, col_indices, row_indices, vals, ubs, lbs, minacts, maxacts,
                                            maxactdeltas);

      change_found = sequentialPropagationRound<datatype>
              (
                      n_cons, n_vars, col_indices, row_indices, csc_col_ptrs, csc_row_indices, vals, lhss, rhss,
                      lbs, ubs, vartypes, minacts, maxacts, maxactdeltas, consmarked, RECOMPUTE_ACTS_FALSE
              );

   }

   VERBOSE_CALL(measureTime("\n cpu_seq", start, std::chrono::steady_clock::now()));
   VERBOSE_CALL(printf("cpu_seq num rounds: %d\n", prop_round-1));

   free(minacts);
   free(maxacts);
   free(maxactdeltas);
   free(consmarked);
}


template<class datatype>
void sequentialPropagate
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
                const int *vartypes
        ) {
   consistify_var_bounds(n_vars, lbs, ubs, vartypes);

   datatype *minacts = (datatype *) calloc(n_cons, sizeof(datatype));
   datatype *maxacts = (datatype *) calloc(n_cons, sizeof(datatype));
   datatype *maxactdeltas = (datatype *) calloc(n_cons, sizeof(datatype));
   int *consmarked = (int *) calloc(n_cons, sizeof(int));

   // all cons marked for propagation in the first round
   for (int i = 0; i < n_cons; i++)
      consmarked[i] = 1;

   // make sure var bounds are consistent - i.e. no integer vars have decimal values.
   consistify_var_bounds(n_vars, lbs, ubs, vartypes);

#ifdef VERBOSE
   auto start = std::chrono::steady_clock::now();
#endif

#ifdef CALC_PROGRESS
  datatype* oldlbs = (datatype*)SAFEMALLOC(n_vars * sizeof(datatype));
  datatype* oldubs = (datatype*)SAFEMALLOC(n_vars * sizeof(datatype));
#endif

   VERBOSE_CALL(printf("\ncpu_seq execution start..."));

   bool change_found = true;
   int prop_round;
   for (prop_round = 1; prop_round <= MAX_NUM_ROUNDS && change_found; prop_round++)  // maxnumrounds = 100
   {
      DEBUG_CALL( printf("\nPropagation round: %d\n\n", prop_round) );
      //    VERBOSE_CALL( countPrintNumMarkedCons<int>(n_cons, consmarked) );
      CALC_PROGRESS_CALL( memcpy(oldlbs, lbs, n_vars * sizeof(datatype)) );
      CALC_PROGRESS_CALL( memcpy(oldubs, ubs, n_vars * sizeof(datatype)) );

      change_found = sequentialPropagationRound<datatype>
              (
                      n_cons, n_vars, col_indices, row_indices, csc_col_ptrs, csc_row_indices, vals, lhss, rhss,
                      lbs, ubs, vartypes, minacts, maxacts, maxactdeltas, consmarked, RECOMPUTE_ACTS_TRUE
              );

      CALC_PROGRESS_CALL(
              printf("\nround %d total score: %.10f",
                      prop_round, calcLocalProgressMeasureSeq(n_vars, oldubs, oldlbs, ubs, lbs))
              );
   }

   VERBOSE_CALL(printf("\ncpu_seq propagation done. Num rounds: %d\n", prop_round-1));
   VERBOSE_CALL(measureTime("cpu_seq", start, std::chrono::steady_clock::now()));

   free(minacts);
   free(maxacts);
   free(maxactdeltas);
   free(consmarked);
   CALC_PROGRESS_CALL( free(oldlbs) );
   CALC_PROGRESS_CALL( free(oldubs) );
}

#endif