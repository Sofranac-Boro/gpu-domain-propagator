
#ifndef __GPUPROPAGATOR_SEQUENTIAL_CUH__
#define __GPUPROPAGATOR_SEQUENTIAL_CUH__


#include "../propagation_methods.cuh"
#include "../misc.h"
#include "../params.h"
#include "../kernels/csr_data.h"

// using namespace Eigen;

#define RECOMPUTE_ACTS_TRUE true
#define RECOMPUTE_ACTS_FALSE false


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
                int *minacts_inf,
                int *maxacts_inf,
                datatype *maxactdeltas,
                int *consmarked,
                int *consmarked_nextround,
                const bool recomputeActs
        ) {

   datatype coeff;
   bool isVarCont;
   datatype rhs;
   datatype lhs;
   bool change_found = false;
   int val_idx;
   int varidx;

   std::fill(consmarked_nextround, consmarked_nextround + n_cons, 0);
   // printf("\nentered sequential propagation round\n");

   for (int considx = 0; considx < n_cons; considx++) {
   // for (int considx = 0; considx < 1; considx++) {
      // printf("\nentered into the first for loop\n");
      if (consmarked[considx] == 1) {
         if (recomputeActs) {
            ActivitiesTuple activities = computeActivities(considx, col_indices, row_indices, vals, ubs, lbs);
            minacts[considx] = activities.minact;
            maxacts[considx] = activities.maxact;
            minacts_inf[considx] = activities.minact_inf;
            maxacts_inf[considx] = activities.maxact_inf;
            maxactdeltas[considx] = activities.maxactdelta;
            // printf("entered into if conditon\n");
         }

         //DEBUG_CALL( printf("cons %d: minact: %.5f, maxact: %.5f\n", considx, minacts[considx], maxacts[considx]) );

         rhs = rhss[considx];
         lhs = lhss[considx];
         // printf("\nminacts\n");
         // for(int j =  0; j<n_cons;j++)
         // {
         //    printf("%d %f, ",j,minacts[j]);
         //    if (j % 10 == 0){ printf("\n"); }
         // }
         // printf("\nmaxacts\n");
         // for(int k =  0; k<n_cons;k++)
         // {
         //    printf("%d %f, ",k,maxacts[k]);
         //    if (k % 10 == 0){ printf("\n"); }
         // }

         // for(int i =  0; i<n_cons;i++)
         // {
         //    printf("\nminacts %f\n",minacts[considx]);
         // }
         FOLLOW_CONS_CALL(considx,
                          printf("\ncons %d: minact:  %9.2e, maxact: %9.2e, minact_inf: %d, maxact_inf: %d, lhs: %9.2e, rhs: %9.2e\n",
                                 considx, minacts[considx], maxacts[considx], minacts_inf[considx],
                                 maxacts_inf[considx], lhs, rhs));
         
         if (canConsBeTightened(minacts[considx], maxacts[considx], minacts_inf[considx], maxacts_inf[considx], lhs,
                                rhs, maxactdeltas[considx])) {
            printf("\nconsidx (i-1) %d \n",considx);
            printf("\nlhs %f \n",lhs);
            printf("\nrhs %f \n",rhs);
            // printf("\nminacts[considx] %f \n",minacts[considx]);
            // printf("\nmaxacts[considx] %f \n",maxacts[considx]);
            // printf("\nminacts_inf[considx] %f \n",minacts_inf[considx]);
            // printf("\nmaxacts_inf[considx] %f \n",maxacts_inf[considx]);
            // printf("\nmaxactdeltas[considx] %f \n",maxactdeltas[considx]);
            int num_vars_in_cons = row_indices[considx + 1] - row_indices[considx];
            printf("num_vars_in_cons %d \n",num_vars_in_cons);
            for (int var = 0; var < num_vars_in_cons; var++) {
               printf("\nvar %d \n",var);
               val_idx = row_indices[considx] + var;
               printf("val_idx %d \n",val_idx);
               varidx = col_indices[val_idx];
               printf("varidx %d \n",varidx);
               coeff = vals[val_idx];
               printf("coeff %f \n",coeff);
               isVarCont = vartypes[varidx] == GDP_CONTINUOUS;
               printf("%s \n",isVarCont ? "true" : "false");
               NewBounds<datatype> newbds = tightenVariable<datatype>
                       (
                               coeff, lhs, rhs, minacts[considx], maxacts[considx], minacts_inf[considx],
                               maxacts_inf[considx], isVarCont, lbs[varidx], ubs[varidx]
                       );

               if (newbds.lb.is_tightened) {
                  FOLLOW_VAR_CALL(varidx,
                                  printf("cpu_seq lb change found: varidx: %7d, considx: %7d, lhs: %9.2e, rhs: %9.2e, coeff: %9.2e, minact: %9.7e, maxact: %9.7e, num_minact_inf: %d,"
                                         " num_maxact_inf: %d, oldlb: %9.2e, oldub: %9.2e, newlb: %9.2e\n",
                                         varidx, considx, lhs, rhs, coeff, minacts[considx], maxacts[considx],
                                         minacts_inf[considx], maxacts_inf[considx], lbs[varidx], ubs[varidx],
                                         newbds.lb.newb)
                  );
                  printf("lb at %d tightened to %f \n",varidx,newbds.lb.newb);
                  
                  lbs[varidx] = newbds.lb.newb;
                  assert(EPSLE(lbs[varidx], ubs[varidx]));

               }

               if (newbds.ub.is_tightened) {
                  FOLLOW_VAR_CALL(varidx,
                                  printf("cpu_seq ub change found: varidx: %7d, considx: %7d, lhs: %9.2e, rhs: %9.2e, coeff: %9.2e, minact: %9.7e, maxact: %9.7e, num_minact_inf: %d,"
                                         " num_maxact_inf: %d, oldlb(or new): %9.2e, oldub: %9.2e, newub: %9.2e\n",
                                         varidx, considx, lhs, rhs, coeff, minacts[considx], maxacts[considx],
                                         minacts_inf[considx], maxacts_inf[considx], lbs[varidx], ubs[varidx],
                                         newbds.ub.newb)
                  );
                  printf("ub at %d has been tightened to %f \n",varidx, newbds.ub.newb);
                  ubs[varidx] = newbds.ub.newb;

                  assert(EPSLE(lbs[varidx], ubs[varidx]));
               }

               if (newbds.ub.is_tightened || newbds.lb.is_tightened) {
                  change_found = true;
                  markConstraints(varidx, csc_col_ptrs, csc_row_indices, consmarked_nextround);
               }
            }
         }
      }
   }

   // copy data from consmarked_nextround into consmarked
   memcpy(consmarked, consmarked_nextround, n_cons * sizeof(int));
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
   int *minacts_inf = (int *) calloc(n_cons, sizeof(int));
   int *maxacts_inf = (int *) calloc(n_cons, sizeof(int));
   datatype *maxactdeltas = (datatype *) calloc(n_cons, sizeof(datatype));
   int *consmarked = (int *) calloc(n_cons, sizeof(int));
   int *consmarked_nextround = (int *) calloc(n_cons, sizeof(int));

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
                                            minacts_inf, maxacts_inf,
                                            maxactdeltas);

      change_found = sequentialPropagationRound<datatype>
              (
                      n_cons, n_vars, col_indices, row_indices, csc_col_ptrs, csc_row_indices, vals, lhss, rhss,
                      lbs, ubs, vartypes, minacts, maxacts, minacts_inf, maxacts_inf, maxactdeltas, consmarked,
                      consmarked_nextround, RECOMPUTE_ACTS_FALSE
              );

   }

   VERBOSE_CALL(printf("\ncpu_seq_dis propagation done. Num rounds: %d\n", prop_round - 1));
   VERBOSE_CALL(measureTime("cpu_seq_dis", start, std::chrono::steady_clock::now()));

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

   // printf("n_cons %d\n",n_cons);
   // printf("n_vars %d\n",n_vars);
   // printf("nnz %d\n",nnz);
   // printf("\nprinting vals : \n");
   // for (int i = 0; i< nnz; i++)
   // {

   // printf("%f ",*(vals+i));
   // }
   // printf("\nprinting column indices : \n");
   // for (int i = 0; i< nnz; i++)
   // {

   // printf("%d ",*(col_indices+i));
   // }
   // printf("\nprinting row pointers : \n");
   // for (int i = 0; i< n_cons+1; i++)
   // {
   // printf("%d ",*(row_indices+i));

   // }
   // // printf("\n*vals %f\n",*vals);
   // printf("\nprinting lhss : \n");
   // for (int i = 0; i< n_cons; i++)
   // {
   //    printf("%f ",*(lhss+i));
   // }
   // printf("\nprinting rhss : \n");
   // for (int i = 0; i< n_cons; i++)
   // {
   //    printf("%f ",*(rhss+i));
   // }
   // printf("\nprinting lbs : \n");
   // for (int i = 0; i< n_vars; i++)
   // {
   //    printf("%f ",*(lbs+i));
   // }
   // printf("\nprinting ubs : \n");
   // for (int i = 0; i< n_vars; i++)
   // {
   //    printf("%f ",*(ubs+i));

   // }
   // printf("\nprinting vartypes : \n");
   // for (int i = 0; i< n_vars; i++)
   // {
   //    printf("%d ",*(vartypes+i));

   // }
   if (n_cons == 0 || n_vars == 0 || nnz == 0) {
      printf("propagation of 0 size problem. Nothing to propagate.\n");
      return GDP_OKAY;
   }

   DEBUG_CALL(checkInput<datatype>(n_cons, n_vars, row_indices[n_cons], vals, lhss, rhss, lbs, ubs, vartypes));

   // get csc format computed
   datatype *csc_vals = (datatype *) SAFEMALLOC(nnz * sizeof(datatype));
   int *csc_row_indices = (int *) SAFEMALLOC(nnz * sizeof(int));
   int *csc_col_ptrs = (int *) SAFEMALLOC((n_vars + 1) * sizeof(int));
   // printf("\nconverting csr to csc\n");
   // printf("printing length of row indices\n");
   // int row_inst[] = &(*row_indices);
   // printf(sizeof(row_inst)/sizeof(row_inst[0]));
   // for (int i = 0; i< n_cons; i++)
   // {
   //    printf("%d, ",*(row_indices+i));
   // }
   // printf("\n");

   // printf("\nprinting csr values in sequential_propagator.h\n");
   // for (int i = 0;i < nnz; i++)
   // {
   //    printf("%f, ",*(vals+i));
   // }

   // csr_to_csc(n_cons,n_vars,nnz,col_indices,row_indices,csc_col_ptrs,csc_row_indices,csc_vals,vals);

   sparseMatrixConverter<datatype> matConverter;
   matConverter.convertToCSC(n_cons,n_vars,nnz,vals,col_indices,row_indices,csc_vals,csc_row_indices,csc_col_ptrs);
   // int length_of_ptrs = ARRAY_SIZE(row_indices);
   // mat.convertToCSC();
   // mat.convertToCSC(n_cons,n_vars,nnz,vals,col_indices,row_indices,csc_vals,csc_row_indices,csc_col_ptrs);
   // printf("\nprinting csc row indices\n");
   // for (int i = 0; i<nnz; i++)
   // {
   //    printf("%d, ",*(csc_row_indices+i));
   // }

   // printf("\nprinting csc coloumn pointers\n");

   // for (int i = 0; i < n_vars+1; i++) {
   //    printf("%d, ",*(csc_col_ptrs+i));
   // }

   // printf("\nprinting csc values\n");
   // for (int i = 0;i < nnz; i++)
   // {
   //    printf("%f, ",*(csc_vals+i));
   // }

   
   datatype *minacts = (datatype *) calloc(n_cons, sizeof(datatype));
   datatype *maxacts = (datatype *) calloc(n_cons, sizeof(datatype));
   int *minacts_inf = (int *) calloc(n_cons, sizeof(int));
   int *maxacts_inf = (int *) calloc(n_cons, sizeof(int));
   datatype *maxactdeltas = (datatype *) calloc(n_cons, sizeof(datatype));
   int *consmarked = (int *) calloc(n_cons, sizeof(int));
   int *consmarked_nextround = (int *) calloc(n_cons, sizeof(int));

   // all cons marked for propagation in the first round
   for (int i = 0; i < n_cons; i++)
      consmarked[i] = 1;

#ifdef VERBOSE
   auto start = std::chrono::steady_clock::now();
#endif

   VERBOSE_CALL(printf(
           "\ncpu_seq execution start... Datatype: %s, MAXNUMROUNDS: %d",
           getDatatypeName<datatype>(), MAX_NUM_ROUNDS));
   VERBOSE_CALL_2(printf("\n"));

   bool change_found = true;
   int prop_round;
   printf("entered in the method GDP_Retcode sequentialPropagate");

   for (prop_round = 1; prop_round <= MAX_NUM_ROUNDS && change_found; prop_round++)  // maxnumrounds = 100
   {
      VERBOSE_CALL_2(printf("Propagation round: %d, ", prop_round));
      FOLLOW_VAR_CALL(FOLLOW_VAR,
                      printf("cpu_seq varidx %d bounds beofre round %d: lb: %9.2e, ub: %9.2e\n", FOLLOW_VAR, prop_round,
                             lbs[FOLLOW_VAR],
                             ubs[FOLLOW_VAR]));

      //printf(change_found);
      change_found = sequentialPropagationRound<datatype>
              (
                      n_cons, n_vars, col_indices, row_indices, csc_col_ptrs, csc_row_indices, vals, lhss, rhss,
                      lbs, ubs, vartypes, minacts, maxacts, minacts_inf, maxacts_inf, maxactdeltas, consmarked,
                      consmarked_nextround, RECOMPUTE_ACTS_TRUE
              );

      FOLLOW_VAR_CALL(FOLLOW_VAR,
                      printf("cpu_seq varidx %d bounds after round %d: lb: %9.2e, ub: %9.2e\n", FOLLOW_VAR, prop_round,
                             lbs[FOLLOW_VAR], ubs[FOLLOW_VAR]));
      VERBOSE_CALL_2(measureTime("cpu_seq", start, std::chrono::steady_clock::now()));
   }

   VERBOSE_CALL(printf("\ncpu_seq propagation done. Num rounds: %d\n", prop_round - 1));
   VERBOSE_CALL(measureTime("cpu_seq", start, std::chrono::steady_clock::now()));

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

   return GDP_OKAY;
}

#endif
