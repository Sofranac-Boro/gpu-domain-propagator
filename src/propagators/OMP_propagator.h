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
                const GDP_VARTYPE *vartypes,
                datatype *minacts,
                datatype *maxacts,
                int* minacts_inf,
                int* maxacts_inf,
                datatype *maxactdeltas,
                int *consmarked,
                omp_lock_t *locks,
                const int round
        ) {

   datatype coeff;
   bool isVarCont;
   datatype slack;
   datatype surplus;

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
      minacts_inf[considx] = activities.minact_inf;
      maxacts_inf[considx] = activities.maxact_inf;
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
            isVarCont = vartypes[var_idx] == GDP_CONTINUOUS;

            omp_set_lock(&(locks[var_idx]));
            // Experiments show that tasks perform worse. This is porbably because they are expensive to manage under
            // the hood, and there is little computation in each task to justify it
            // #pragma omp task depend(inout: ubs[var_idx], lbs[var_idx])
            {
               NewBounds newbds = tightenVariable<datatype>
                       (
                               coeff, lhss[considx], rhss[considx], minacts[considx], maxacts[considx], minacts_inf[considx], maxacts_inf[considx], isVarCont,
                               var_idx, val_idx,
                               csc_col_ptrs, csc_row_indices, lbs, ubs
                       );

               if (newbds.lb.is_tightened)
               {
                  lbs[var_idx] = newbds.lb.newb;
               }

               if (newbds.ub.is_tightened)
               {
                  ubs[var_idx] = newbds.ub.newb;
               }

               if (newbds.ub.is_tightened || newbds.lb.is_tightened)
               {
                  change_found = true;
                  markConstraints(var_idx, csc_col_ptrs, csc_row_indices, consmarked_nextround);
               }
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
GDP_Retcode fullOMPPropagate
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
   int *minacts_inf = (int *) calloc(n_cons, sizeof(int));
   int *maxacts_inf = (int *) calloc(n_cons, sizeof(int));
   datatype *maxactdeltas = (datatype *) calloc(n_cons, sizeof(datatype));
   int *consmarked = (int *) calloc(n_cons, sizeof(int));

   // all cons marked for propagation in the first round
   for (int i = 0; i < n_cons; i++)
      consmarked[i] = 1;

#ifdef VERBOSE
   auto start = std::chrono::steady_clock::now();
#endif

   VERBOSE_CALL(printf("\ncpu_omp execution start... OMP num threads: %d, Datatype: %s, MAXNUMROUNDS: %d\n",
                       SHARED_MEM_THREADS, getDatatypeName<datatype>(), MAX_NUM_ROUNDS));

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
                      lbs, ubs, vartypes, minacts, maxacts, minacts_inf, maxacts_inf, maxactdeltas, consmarked, locks, prop_round
              );
   }

   for (int i = 0; i < n_vars; i++)
      omp_destroy_lock(&(locks[i]));

   VERBOSE_CALL(printf("cpu_omp propagation done. Num rounds: %d\n", prop_round - 1));
   VERBOSE_CALL(measureTime("cpu_omp", start, std::chrono::steady_clock::now()));

   free(minacts);
   free(maxacts);
   free(minacts_inf);
   free(maxacts_inf);
   free(maxactdeltas);
   free(consmarked);
   free(csc_vals);
   free(csc_col_ptrs);
   free(csc_row_indices);

   return GDP_OKAY;
}

#endif