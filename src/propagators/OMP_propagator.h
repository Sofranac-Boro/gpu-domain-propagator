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
                datatype *lbs_local,
                datatype *ubs_local,
                const GDP_VARTYPE *vartypes,
                datatype *minacts,
                datatype *maxacts,
                int* minacts_inf,
                int* maxacts_inf,
                datatype *maxactdeltas,
                int *consmarked,
                omp_lock_t *locks,
                int* consmarked_nextround,
                const int round
        ) {

   datatype coeff;
   bool isVarCont;
   bool change_found = false;
   int val_idx;
   int varidx;
   int num_marked_cons = 0;

   std::fill(consmarked_nextround, consmarked_nextround+n_cons, 0);
   memcpy(lbs_local, lbs, n_vars*sizeof(datatype));
   memcpy(ubs_local, ubs, n_vars*sizeof(datatype));

   // put all indices of marked constraints at the beginning of the array
   for (int i = 0; i < n_cons; i++) {
      if (consmarked[i] == 1) {
         consmarked[num_marked_cons] = i;
         num_marked_cons++;
      }
   }

#pragma omp parallel for default(shared) private(val_idx, varidx, coeff, isVarCont) schedule(static) num_threads(SHARED_MEM_THREADS)
   for (int i = 0; i < num_marked_cons; i++) {
      int considx = consmarked[i];

      ActivitiesTuple activities = computeActivities(considx, col_indices, row_indices, vals, ubs_local, lbs_local);
      minacts[considx] = activities.minact;
      maxacts[considx] = activities.maxact;
      minacts_inf[considx] = activities.minact_inf;
      maxacts_inf[considx] = activities.maxact_inf;
      maxactdeltas[considx] = activities.maxactdelta;

      FOLLOW_CONS_CALL(considx, printf("\ncons %d: minact:  %9.2e, maxact: %9.2e, minact_inf: %d, maxact_inf: %d, lhs: %9.2e, rhs: %9.2e\n",
                                       considx, minacts[considx], maxacts[considx], minacts_inf[considx], maxacts_inf[considx], lhss[considx], rhss[considx]));

      if (canConsBeTightened(minacts[considx], maxacts[considx], minacts_inf[considx], maxacts_inf[considx], lhss[considx], rhss[considx], maxactdeltas[considx])) {
         const int num_vars_in_cons = row_indices[considx + 1] - row_indices[considx];
         //slack = EPSLT(slack, 0.0) ? 0.0 : slack;
         for (int var = 0; var < num_vars_in_cons; var++) {
            val_idx = row_indices[considx] + var;
            varidx = col_indices[val_idx];
            coeff = vals[val_idx];
            isVarCont = vartypes[varidx] == GDP_CONTINUOUS;
            NewBounds<datatype> newbds;

            newbds = tightenVariable<datatype>
                    (
                            coeff, lhss[considx], rhss[considx], minacts[considx], maxacts[considx],
                            minacts_inf[considx], maxacts_inf[considx], isVarCont, lbs_local[varidx], ubs_local[varidx]
                    );

            if (newbds.lb.is_tightened) {
               FOLLOW_VAR_CALL(varidx,
                               printf("cpu_seq lb change found: varidx: %7d, considx: %7d, lhs: %9.2e, rhs: %9.2e, coeff: %9.2e, minact: %9.7e, maxact: %9.7e, num_minact_inf: %d,"
                                      " num_maxact_inf: %d, oldlb_local: %9.2e, oldub_local: %9.2e, newlb: %9.2e\n",
                                      varidx, considx, lhss[considx], rhss[considx], coeff, minacts[considx],
                                      maxacts[considx], minacts_inf[considx], maxacts_inf[considx], lbs_local[varidx],
                                      ubs_local[varidx], newbds.lb.newb)
               );
               omp_set_lock(&(locks[varidx]));
               {
                  if (isLbBetter(lbs[varidx], ubs[varidx], newbds.lb.newb))
                  {
                     lbs[varidx] = newbds.lb.newb;
                     assert(EPSLE(lbs[varidx], ubs[varidx]));
                  }
               }
               omp_unset_lock(&(locks[varidx]));
            }

            if (newbds.ub.is_tightened) {
               FOLLOW_VAR_CALL(varidx,
                               printf("cpu_seq ub change found: varidx: %7d, considx: %7d, lhs: %9.2e, rhs: %9.2e, coeff: %9.2e, minact: %9.7e, maxact: %9.7e, num_minact_inf: %d,"
                                      " num_maxact_inf: %d, oldlb_local(or new): %9.2e, oldub_local: %9.2e, newub: %9.2e\n",
                                      varidx, considx, lhss[considx], rhss[considx], coeff, minacts[considx],
                                      maxacts[considx], minacts_inf[considx], maxacts_inf[considx], lbs_local[varidx],
                                      ubs_local[varidx], newbds.ub.newb)
               );
               omp_set_lock(&(locks[varidx]));
               {
                  if (isUbBetter(lbs[varidx], ubs[varidx], newbds.ub.newb))
                  {
                     ubs[varidx] = newbds.ub.newb;
                     assert(EPSLE(lbs[varidx], ubs[varidx]));
                  }
               }
               omp_unset_lock(&(locks[varidx]));
            }

            if (newbds.ub.is_tightened || newbds.lb.is_tightened) {
               change_found = true;
               markConstraints(varidx, csc_col_ptrs, csc_row_indices, consmarked_nextround);
            }
         }
      }
   }

   // copy data from consmarked_nextround into consmarked
   memcpy(consmarked, consmarked_nextround, n_cons * sizeof(int));
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
   int *consmarked_nextround = (int *) calloc(n_cons, sizeof(int));
   datatype* lbs_local = (datatype*) malloc(n_vars*sizeof(datatype));
   datatype* ubs_local = (datatype*) malloc(n_vars*sizeof(datatype));
   // all cons marked for propagation in the first round
   for (int i = 0; i < n_cons; i++)
      consmarked[i] = 1;

   //initialize omp locks used in propagation rounds:
   omp_lock_t locks[n_vars];
   for (int i = 0; i < n_vars; i++)
      omp_init_lock(&(locks[i]));

#ifdef VERBOSE
   auto start = std::chrono::steady_clock::now();
#endif

   VERBOSE_CALL(printf("\ncpu_omp execution start... OMP num threads: %d, Datatype: %s, MAXNUMROUNDS: %d\n",
                       SHARED_MEM_THREADS, getDatatypeName<datatype>(), MAX_NUM_ROUNDS));

   bool change_found = true;
   int prop_round;
   for (prop_round = 1; prop_round < MAX_NUM_ROUNDS && change_found; prop_round++) {
      VERBOSE_CALL_2(printf("Propagation round: %d, ", prop_round));
      FOLLOW_VAR_CALL(FOLLOW_VAR,
                      printf("cpu_omp varidx %d bounds beofre round %d: lb: %9.2e, ub: %9.2e\n", FOLLOW_VAR, prop_round, lbs[FOLLOW_VAR],
                             ubs[FOLLOW_VAR]));
      change_found = OMPPropagationRound<datatype>
              (
                      n_cons, n_vars, col_indices, row_indices, csc_col_ptrs, csc_row_indices, vals, lhss, rhss,
                      lbs, ubs, lbs_local, ubs_local, vartypes, minacts, maxacts, minacts_inf, maxacts_inf, maxactdeltas, consmarked, locks, consmarked_nextround, prop_round
              );
      FOLLOW_VAR_CALL(FOLLOW_VAR,
                      printf("cpu_omp varidx %d bounds after round %d: lb: %9.2e, ub: %9.2e\n", FOLLOW_VAR, prop_round, lbs[FOLLOW_VAR], ubs[FOLLOW_VAR]));
      VERBOSE_CALL_2( measureTime("cpu_omp", start, std::chrono::steady_clock::now()) );
   }

   VERBOSE_CALL(printf("cpu_omp propagation done. Num rounds: %d\n", prop_round - 1));
   VERBOSE_CALL(measureTime("cpu_omp", start, std::chrono::steady_clock::now()));

   for (int i = 0; i < n_vars; i++)
      omp_destroy_lock(&(locks[i]));

   free(minacts);
   free(maxacts);
   free(minacts_inf);
   free(maxacts_inf);
   free(maxactdeltas);
   free(consmarked);
   free(csc_vals);
   free(csc_col_ptrs);
   free(csc_row_indices);
   free(consmarked_nextround);
   free(lbs_local);
   free(ubs_local);

   return GDP_OKAY;
}

#endif