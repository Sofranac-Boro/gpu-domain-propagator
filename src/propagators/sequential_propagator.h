
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

         //DEBUG_CALL( printf("cons %d: minact: %.5f, maxact: %.5f\n", considx, minacts[considx], maxacts[considx]) );

         rhs = rhss[considx];
         lhs = lhss[considx];
         slack = rhs - minacts[considx];
         surplus = maxacts[considx] - lhs;

         if (canConsBeTightened(slack, surplus, maxactdeltas[considx])) {
            int num_vars_in_cons = row_indices[considx + 1] - row_indices[considx];
            slack = EPSLT(slack, 0.0) ? 0.0 : slack;

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

   DEBUG_CALL(checkInput(n_cons, n_vars, row_indices[n_cons], vals, lhss, rhss, lbs, ubs));
   consistify_var_bounds(n_vars, lbs, ubs, vartypes);

   datatype *minacts = (datatype *) calloc(n_cons, sizeof(datatype));
   datatype *maxacts = (datatype *) calloc(n_cons, sizeof(datatype));
   datatype *maxactdeltas = (datatype *) calloc(n_cons, sizeof(datatype));
   int *consmarked = (int *) calloc(n_cons, sizeof(int));

#ifdef VERBOSE
   auto start = std::chrono::steady_clock::now();
#endif

   VERBOSE_CALL(printf("\ncpu_seq_dis execution start... Params: MAXNUMROUNDS: %d", MAX_NUM_ROUNDS));

   bool change_found = true;
   int prop_round = 0;
   for (prop_round = 1; prop_round <= MAX_NUM_ROUNDS && change_found; prop_round++)  // maxnumrounds = 100
   {
      VERBOSE_CALL_2(printf("\nPropagation round: %d\n\n", prop_round));
      //VERBOSE_CALL( countPrintNumMarkedCons<int>(n_cons, consmarked) );

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

   DEBUG_CALL(checkInput(n_cons, n_vars, row_indices[n_cons], vals, lhss, rhss, lbs, ubs));

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

#ifdef CALC_PROGRESS_REL
   datatype* oldlbs = (datatype*)SAFEMALLOC(n_vars * sizeof(datatype));
   datatype* oldubs = (datatype*)SAFEMALLOC(n_vars * sizeof(datatype));

   int* rel_measure_k = (int*)SAFEMALLOC(sizeof(int));
#endif
#ifdef CALC_PROGRESS_ABS
   datatype* reflbs = (datatype*)SAFEMALLOC(n_vars * sizeof(datatype));
   datatype* refubs = (datatype*)SAFEMALLOC(n_vars * sizeof(datatype));
   memcpy(reflbs, lbs, n_vars * sizeof(datatype));
   memcpy(refubs, ubs, n_vars * sizeof(datatype));
   int* abs_measure_k = (int*)SAFEMALLOC(sizeof(int));
   int* abs_measure_n = (int*)SAFEMALLOC(sizeof(int));
   *abs_measure_k = 0;
   *abs_measure_n = 0;
#endif

   VERBOSE_CALL(printf("\ncpu_seq execution start... Params: MAXNUMROUNDS: %d", MAX_NUM_ROUNDS));

   bool change_found = true;
   int prop_round;
   for (prop_round = 1; prop_round <= MAX_NUM_ROUNDS && change_found; prop_round++)  // maxnumrounds = 100
   {
      VERBOSE_CALL_2(printf("\nPropagation round: %d\n\n", prop_round));
      //    VERBOSE_CALL( countPrintNumMarkedCons<int>(n_cons, consmarked) );
      CALC_PROGRESS_REL_CALL(memcpy(oldlbs, lbs, n_vars * sizeof(datatype)));
      CALC_PROGRESS_REL_CALL(memcpy(oldubs, ubs, n_vars * sizeof(datatype)));

      change_found = sequentialPropagationRound<datatype>
              (
                      n_cons, n_vars, col_indices, row_indices, csc_col_ptrs, csc_row_indices, vals, lhss, rhss,
                      lbs, ubs, vartypes, minacts, maxacts, maxactdeltas, consmarked, RECOMPUTE_ACTS_TRUE
              );

#ifdef CALC_PROGRESS_REL
      *rel_measure_k = 0;
      double score = calcRelProgressMeasureSeq(n_vars, oldlbs, oldubs, lbs, ubs, rel_measure_k);
      printf("\nround %d total relative score: %.10f, k=%d", prop_round, score, *rel_measure_k);
#endif
#ifdef CALC_PROGRESS_ABS
      double abs_score = calcAbsProgressMeasureSeq(n_vars, reflbs, refubs, lbs, ubs, abs_measure_k, abs_measure_n);
      printf("\nround %d total absolute score: %.10f, k=%d, n=%d", prop_round, abs_score, *abs_measure_k, *abs_measure_n);
#endif
   }

   VERBOSE_CALL(printf("\ncpu_seq propagation done. Num rounds: %d\n", prop_round - 1));
   VERBOSE_CALL(measureTime("cpu_seq", start, std::chrono::steady_clock::now()));

   free(minacts);
   free(maxacts);
   free(maxactdeltas);
   free(consmarked);
   CALC_PROGRESS_REL_CALL(free(oldlbs));
   CALC_PROGRESS_REL_CALL(free(oldubs));
   CALC_PROGRESS_REL_CALL(free(rel_measure_k));
}

#endif