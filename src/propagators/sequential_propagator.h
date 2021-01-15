
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

               if (newbds.lb.is_tightened)
               {
                  FOLLOW_VAR_CALL(varidx,
                                  printf("cpu_seq lb change found: varidx: %7d, considx: %7d, oldlb: %9.2e, newlb: %9.2e\n", varidx, considx, lbs[varidx],
                                         newbds.lb.newb));
                  lbs[varidx] = newbds.lb.newb;

               }

               if (newbds.ub.is_tightened)
               {
                  FOLLOW_VAR_CALL(varidx,
                                  printf("cpu_seq ub change found: varidx: %7d, considx: %7d, oldub: %9.2e, newub: %9.2e\n", varidx, considx, ubs[varidx],
                                         newbds.ub.newb));
                  ubs[varidx] = newbds.ub.newb;
               }

               if (newbds.ub.is_tightened || newbds.lb.is_tightened)
               {
                  change_found = true;
                  markConstraints(varidx, csc_col_ptrs, csc_row_indices, consmarked_nextround);
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
GDP_Retcode sequentialPropagateDisjoint
        (
                const int n_cons,
                const int n_vars,
                const int nnz,
                const int *col_indices,
                const int *row_indices,
                const datatype *vals,
                const datatype *lhss,
                const datatype *rhss,
                datatype *lbs,
                datatype *ubs,
                const GDP_VARTYPE *vartypes
        ) {
   if (n_cons == 0 || n_vars == 0 || nnz == 0) {
      printf("propagation of 0 size problem. Nothing to propagate.\n");
      return GDP_OKAY;
   }

   DEBUG_CALL(checkInput<datatype>(n_cons, n_vars, row_indices[n_cons], vals, lhss, rhss, lbs, ubs, vartypes));

   datatype *csc_vals = (datatype *) SAFEMALLOC(nnz * sizeof(datatype));
   int *csc_row_indices = (int *) SAFEMALLOC(nnz * sizeof(int));
   int *csc_col_ptrs = (int *) SAFEMALLOC((n_vars + 1) * sizeof(int));

   csr_to_csc(n_cons, n_vars, nnz, col_indices, row_indices, csc_col_ptrs, csc_row_indices, csc_vals, vals);

   datatype *minacts = (datatype *) calloc(n_cons, sizeof(datatype));
   datatype *maxacts = (datatype *) calloc(n_cons, sizeof(datatype));
   datatype *maxactdeltas = (datatype *) calloc(n_cons, sizeof(datatype));
   int *consmarked = (int *) calloc(n_cons, sizeof(int));

   // all cons marked for propagation in the first round
   for (int i = 0; i < n_cons; i++)
      consmarked[i] = 1;

#ifdef VERBOSE
   auto start = std::chrono::steady_clock::now();
#endif

   VERBOSE_CALL(printf("\ncpu_seq_dis execution start... MAXNUMROUNDS: %d", MAX_NUM_ROUNDS));

   bool change_found = true;
   int prop_round;
   for (prop_round = 1; prop_round <= MAX_NUM_ROUNDS && change_found; prop_round++)  // maxnumrounds = 100
   {
      VERBOSE_CALL_2(printf("Propagation round: %d\n", prop_round));

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
   free(csc_vals);
   free(csc_col_ptrs);
   free(csc_row_indices);

   return GDP_OKAY;
}


template<class datatype>
GDP_Retcode sequentialPropagate
        (
                const int n_cons,
                const int n_vars,
                const int nnz,
                const int *col_indices,
                const int *row_indices,
                const datatype *vals,
                const datatype *lhss,
                const datatype *rhss,
                datatype *lbs,
                datatype *ubs,
                const GDP_VARTYPE *vartypes
        ) {
   if (n_cons == 0 || n_vars == 0 || nnz == 0) {
      printf("propagation of 0 size problem. Nothing to propagate.\n");
      return GDP_OKAY;
   }

   DEBUG_CALL(checkInput<datatype>(n_cons, n_vars, row_indices[n_cons], vals, lhss, rhss, lbs, ubs, vartypes));

   // get csc format computed
   datatype *csc_vals = (datatype *) SAFEMALLOC(nnz * sizeof(datatype));
   int *csc_row_indices = (int *) SAFEMALLOC(nnz * sizeof(int));
   int *csc_col_ptrs = (int *) SAFEMALLOC((n_vars + 1) * sizeof(int));

   csr_to_csc(n_cons, n_vars, nnz, col_indices, row_indices, csc_col_ptrs, csc_row_indices, csc_vals, vals);

   datatype *minacts = (datatype *) calloc(n_cons, sizeof(datatype));
   datatype *maxacts = (datatype *) calloc(n_cons, sizeof(datatype));
   datatype *maxactdeltas = (datatype *) calloc(n_cons, sizeof(datatype));
   int *consmarked = (int *) calloc(n_cons, sizeof(int));

   // all cons marked for propagation in the first round
   for (int i = 0; i < n_cons; i++)
      consmarked[i] = 1;

#ifdef VERBOSE
   auto start = std::chrono::steady_clock::now();
#endif

   VERBOSE_CALL(printf(
           "\ncpu_seq execution start... Datatype: %s, MAXNUMROUNDS: %d",
           getDatatypeName<datatype>(), MAX_NUM_ROUNDS));
   VERBOSE_CALL_2( printf("\n") );

   bool change_found = true;
   int prop_round;
   for (prop_round = 1; prop_round <= MAX_NUM_ROUNDS && change_found; prop_round++)  // maxnumrounds = 100
   {
      FOLLOW_VAR_CALL(FOLLOW_VAR,
                      printf("cpu_seq varidx: %7d bounds beofre round: %7d: lb: %9.2e, ub: %9.2e\n", FOLLOW_VAR, prop_round, lbs[FOLLOW_VAR],
                             ubs[FOLLOW_VAR]));

      change_found = sequentialPropagationRound<datatype>
              (
                      n_cons, n_vars, col_indices, row_indices, csc_col_ptrs, csc_row_indices, vals, lhss, rhss,
                      lbs, ubs, vartypes, minacts, maxacts, maxactdeltas, consmarked, RECOMPUTE_ACTS_TRUE
              );

      FOLLOW_VAR_CALL(FOLLOW_VAR,
                      printf("cpu_seq varidx %7d bounds after round %7d: lb: %9.2e, ub: %9.2e\n", FOLLOW_VAR, prop_round, lbs[FOLLOW_VAR], ubs[FOLLOW_VAR]));

      VERBOSE_CALL_2(printf("Propagation round: %d, ", prop_round));
      VERBOSE_CALL_2( measureTime("cpu_seq", start, std::chrono::steady_clock::now()) );
   }

   VERBOSE_CALL(printf("\ncpu_seq propagation done. Num rounds: %d\n", prop_round - 1));
   VERBOSE_CALL(measureTime("cpu_seq", start, std::chrono::steady_clock::now()));

   free(minacts);
   free(maxacts);
   free(maxactdeltas);
   free(consmarked);
   free(csc_vals);
   free(csc_col_ptrs);
   free(csc_row_indices);

   return GDP_OKAY;
}

#endif