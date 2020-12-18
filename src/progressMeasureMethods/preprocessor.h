#ifndef GPU_DOMAIN_PROPAGATOR_PREPROCESSOR_H
#define GPU_DOMAIN_PROPAGATOR_PREPROCESSOR_H

#include "../propagators/sequential_propagator.h"

template<class datatype>
bool preprocessorPropagationRound
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
                datatype* lbs_original,
                datatype* ubs_original,
                const GDP_VARTYPE *vartypes,
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
   int varidx;

   int *consmarked_nextround = (int *) calloc(n_cons, sizeof(int));

   // record current bounds so that we have a reference to the starting bounds of the round
   memcpy(lbs_original, lbs, n_vars * sizeof(datatype));
   memcpy(ubs_original, ubs, n_vars * sizeof(datatype));

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
               varidx = col_indices[val_idx];
               coeff = vals[val_idx];

               isVarCont = vartypes[varidx] == GDP_CONTINUOUS;

               NewBounds newbds = tightenVariable<datatype>
                       (
                               coeff, lhs, rhs, minacts[considx], maxacts[considx], isVarCont, varidx, val_idx,
                               csc_col_ptrs, csc_row_indices, lbs, ubs
                       );


               // if the bounds was infinite before, and the new bound is finite
               if (newbds.lb.is_tightened && EPSLE(lbs_original[varidx], -GDP_INF) && EPSGT(newbds.lb.newb, -GDP_INF))
               {
                  // it could happen that some other constraint in the system already found a finite tightening for this var.
                  // In this case, only update if the new finite bound is "worse"
                  if ( EPSLE(lbs[varidx], -GDP_INF) || ( EPSGT(lbs[varidx], -GDP_INF) && EPSLT(newbds.lb.newb, lbs[varidx]) ) )
                  {
                     FOLLOW_VAR_CALL(varidx,
                                     printf("preprocessor lb change found: varidx: %7d, oldlb: %9.2e, newlb: %9.2e\n", varidx, lbs[varidx],
                                            newbds.lb.newb));
                     lbs[varidx] = newbds.lb.newb;
                     change_found = true;
                     markConstraints(varidx, csc_col_ptrs, csc_row_indices, consmarked_nextround);
                  }
               }

               // if the bounds was infinite before, and the new bound is finite
               if (newbds.ub.is_tightened && EPSGE(ubs_original[varidx], GDP_INF) && EPSLT(newbds.ub.newb, GDP_INF))
               {
                  // it could happen that some other constraint in the system already found a finite tightening for this var.
                  // In this case, only update if the new finite bound is "worse"
                  if ( EPSGE(ubs[varidx], GDP_INF) || ( EPSLT(ubs[varidx], GDP_INF) && EPSGT(newbds.ub.newb, ubs[varidx]) ) )
                  {
                     FOLLOW_VAR_CALL(varidx,
                                     printf("preprocessor ub change found: varidx: %7d, oldub: %9.2e, newub: %9.2e\n", varidx, ubs[varidx],
                                            newbds.ub.newb));
                     ubs[varidx] = newbds.ub.newb;
                     change_found = true;
                     markConstraints(varidx, csc_col_ptrs, csc_row_indices, consmarked_nextround);
                  }
               }
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
void executePreprocessor
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

   datatype* lbs_original = (datatype*)SAFEMALLOC(n_vars * sizeof(datatype));
   datatype* ubs_original = (datatype*)SAFEMALLOC(n_vars * sizeof(datatype));

   // all cons marked for propagation in the first round
   for (int i = 0; i < n_cons; i++)
      consmarked[i] = 1;

#ifdef VERBOSE
   auto start = std::chrono::steady_clock::now();
#endif

   VERBOSE_CALL(printf("\ncpu_seq_dis execution start... Params: MAXNUMROUNDS: %d", MAX_NUM_ROUNDS));

   bool change_found = true;
   int prop_round;
   for (prop_round = 1; prop_round <= MAX_NUM_ROUNDS && change_found; prop_round++)  // maxnumrounds = 100
   {
      VERBOSE_CALL_2(printf("\nPropagation round: %d", prop_round));

      sequentialComputeActivities<datatype>(n_cons, col_indices, row_indices, vals, ubs, lbs, minacts, maxacts,
                                            maxactdeltas);

      FOLLOW_VAR_CALL(FOLLOW_VAR,
                      printf("preprocessor varidx: %7d bounds before round: %7d: lb: %9.2e, ub: %9.2e\n", FOLLOW_VAR, prop_round, lbs[FOLLOW_VAR],
                             ubs[FOLLOW_VAR]));

      change_found = preprocessorPropagationRound<datatype>
              (
                      n_cons, n_vars, col_indices, row_indices, csc_col_ptrs, csc_row_indices, vals, lhss, rhss,
                      lbs, ubs, lbs_original, ubs_original, vartypes, minacts, maxacts, maxactdeltas, consmarked, RECOMPUTE_ACTS_FALSE
              );

      FOLLOW_VAR_CALL(FOLLOW_VAR,
                      printf("preprocessor varidx: %7d bounds after round: %7d: lb: %9.2e, ub: %9.2e\n", FOLLOW_VAR, prop_round, lbs[FOLLOW_VAR],
                             ubs[FOLLOW_VAR]));
   }

   VERBOSE_CALL(printf("\ncpu_seq_dis propagation done. Num rounds: %d\n", prop_round - 1));
   VERBOSE_CALL(measureTime("cpu_seq_dis", start, std::chrono::steady_clock::now()));

   free(minacts);
   free(maxacts);
   free(maxactdeltas);
   free(consmarked);
   free(lbs_original);
   free(ubs_original);
}
#endif //GPU_DOMAIN_PROPAGATOR_PREPROCESSOR_H
