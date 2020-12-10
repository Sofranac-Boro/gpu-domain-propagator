#ifndef __GPUPROPAGATOR_SHAREDMEM_CUH__
#define __GPUPROPAGATOR_SHAREDMEM_CUH__

#include "../propagation_methods.cuh"
#include "../cuda_def.cuh"
#include "../misc.h"
#include "../params.h"
#include <omp.h>

template<class datatype>
bool OMPPropagationRound
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
                omp_lock_t *locks,
                const int round
        ) {

   double coeff;
   bool isVarCont;
   double slack;
   double surplus;

   bool change_found = false;
   int val_idx;
   int var_idx;
   int num_marked_cons = 0;

   int *consmarked_nextround = (int *) calloc(n_cons, sizeof(int));

   // put all indices of marked constraints at the beginning of the array
   for (int i = 0; i < n_cons; i++) {
      if (consmarked[i] == 1) {
         consmarked[num_marked_cons] = i;
         num_marked_cons++;
      }
   }

#pragma omp parallel for default(shared) private(slack, surplus, val_idx, var_idx, coeff, isVarCont) schedule(static) num_threads(SHARED_MEM_THREADS)
   for (int i = 0; i < num_marked_cons; i++) {
      int considx = consmarked[i];

      ActivitiesTuple activities = computeActivities(considx, col_indices, row_indices, vals, ubs, lbs);
      minacts[considx] = activities.minact;
      maxacts[considx] = activities.maxact;
      maxactdeltas[considx] = activities.maxactdelta;

      slack = rhss[considx] - minacts[considx];
      surplus = maxacts[considx] - lhss[considx];

      if (canConsBeTightened(slack, surplus, maxactdeltas[considx])) {
         int num_vars_in_cons = row_indices[considx + 1] - row_indices[considx];
         slack = EPSLT(slack, 0.0) ? 0.0 : slack;
         for (int var = 0; var < num_vars_in_cons; var++) {
            val_idx = row_indices[considx] + var;
            var_idx = col_indices[val_idx];

            coeff = vals[val_idx];
            isVarCont = vartypes[var_idx] == 3;

            omp_set_lock(&(locks[var_idx]));
            // Experiments show that tasks perform worse. This is porbably because they are expensive to manage under
            // the hood, and there is little computation in each task to justify it
            // #pragma omp task depend(inout: ubs[var_idx], lbs[var_idx])
            {
               bool tightened = tightenVariable<datatype>
                       (
                               coeff, lhss[considx], rhss[considx], minacts[considx], maxacts[considx], isVarCont,
                               var_idx, val_idx,
                               csc_col_ptrs, csc_row_indices, consmarked_nextround, lbs, ubs
                       );

               change_found = tightened ? tightened : change_found;
            }
            omp_unset_lock(&(locks[var_idx]));
         }
      }
   }

   // copy data from consmarked_nextround into consmarked
   memcpy(consmarked, consmarked_nextround, n_cons * sizeof(int));
   free(consmarked_nextround);
   return change_found;
}

template<class datatype>
void fullOMPPropagate
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

   VERBOSE_CALL(printf("\ncpu_omp execution start with OMP num threads: %d, MAXNUMROUNDS: %d\n", SHARED_MEM_THREADS,
                       MAX_NUM_ROUNDS));

   //initialize omp locks used in propagation rounds:
   omp_lock_t locks[n_vars];
   for (int i = 0; i < n_vars; i++)
      omp_init_lock(&(locks[i]));

   bool change_found = true;
   int prop_round;
   for (prop_round = 1; prop_round < MAX_NUM_ROUNDS && change_found; prop_round++) {
      change_found = OMPPropagationRound<datatype>
              (
                      n_cons, n_vars, col_indices, row_indices, csc_col_ptrs, csc_row_indices, vals, lhss, rhss,
                      lbs, ubs, vartypes, minacts, maxacts, maxactdeltas, consmarked, locks, prop_round
              );
   }

   for (int i = 0; i < n_vars; i++)
      omp_destroy_lock(&(locks[i]));

   VERBOSE_CALL(printf("cpu_omp propagation done. Num rounds: %d\n", prop_round - 1));
   VERBOSE_CALL(measureTime("cpu_omp", start, std::chrono::steady_clock::now()));

   free(minacts);
   free(maxacts);
   free(maxactdeltas);
   free(consmarked);
}

#endif