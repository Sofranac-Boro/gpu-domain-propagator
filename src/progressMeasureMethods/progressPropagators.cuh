#ifndef __GDP_PREGRESS_MEASURE_SEQ_CUH__
#define __GDP_PREGRESS_MEASURE_SEQ_CUH__

#include "../def.h"
#include "../propagators/sequential_propagator.h"
#include "weakest_bounds.h"
#include "../kernels/atomic_kernel.cuh"
#include "progress_function.h"
#include <memory>

template<class datatype>
GDP_Retcode sequentialPropagateWithMeasure
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

   // keep a reference to original bounds. Will need this later.
   datatype *lbs_orig = (datatype *) SAFEMALLOC(n_vars * sizeof(datatype));
   datatype *ubs_orig = (datatype *) SAFEMALLOC(n_vars * sizeof(datatype));
   memcpy(lbs_orig, lbs, n_vars * sizeof(datatype));
   memcpy(ubs_orig, ubs, n_vars * sizeof(datatype));

   // need csc format of A.
   datatype *csc_vals = (datatype *) SAFEMALLOC(nnz * sizeof(datatype));
   int *csc_row_indices = (int *) SAFEMALLOC(nnz * sizeof(int));
   int *csc_col_ptrs = (int *) SAFEMALLOC((n_vars + 1) * sizeof(int));

   csr_to_csc(n_cons, n_vars, nnz, col_indices, row_indices, csc_col_ptrs, csc_row_indices, csc_vals, vals);

   printf("\n=== cpu_seq execution with measure of progress ===\n");
   DEBUG_CALL(checkInput<datatype>(n_cons, n_vars, row_indices[n_cons], vals, lhss, rhss, lbs, ubs, vartypes));

   datatype *lbs_weakest = (datatype *) SAFEMALLOC(n_vars * sizeof(datatype));
   datatype *ubs_weakest = (datatype *) SAFEMALLOC(n_vars * sizeof(datatype));
   datatype *lbs_limit = (datatype *) SAFEMALLOC(n_vars * sizeof(datatype));
   datatype *ubs_limit = (datatype *) SAFEMALLOC(n_vars * sizeof(datatype));
   datatype *lbs_prev = (datatype *) malloc(n_vars * sizeof(datatype));
   datatype *ubs_prev = (datatype *) malloc(n_vars * sizeof(datatype));

   initMeasureData<datatype>
           (n_cons, n_vars, nnz, lbs, ubs, col_indices, row_indices, vals,
            lhss, rhss, vartypes, csc_row_indices, csc_col_ptrs, csc_vals, lbs_weakest, ubs_weakest, lbs_limit, ubs_limit
           );

   const ProgressMeasure<datatype> P_max = calcMaxMeasureValues(n_vars, lbs_orig, ubs_orig, lbs_weakest, ubs_weakest, lbs_limit, ubs_limit); // maximum attainable measure
   int P_inf_total = 0;

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

   VERBOSE_CALL(printf("\n====   Running the cpu_seq with measure  ====\n"));

   bool change_found = true;
   int prop_round;
   for (prop_round = 1; prop_round <= MAX_NUM_ROUNDS && change_found; prop_round++)  // maxnumrounds = 100
   {

      memcpy(lbs_prev, lbs, n_vars * sizeof(datatype));
      memcpy(ubs_prev, ubs, n_vars * sizeof(datatype));

      FOLLOW_VAR_CALL(FOLLOW_VAR,
                      printf("cpu_seq varidx %d bounds beofre round %d: lb: %9.2e, ub: %9.2e\n", FOLLOW_VAR, prop_round,
                             lbs[FOLLOW_VAR],
                             ubs[FOLLOW_VAR]));

      change_found = sequentialPropagationRound<datatype>
              (
                      n_cons, n_vars, col_indices, row_indices, csc_col_ptrs, csc_row_indices, vals, lhss, rhss,
                      lbs, ubs, vartypes, minacts, maxacts, minacts_inf, maxacts_inf, maxactdeltas, consmarked,
                      consmarked_nextround, RECOMPUTE_ACTS_TRUE
              );

      FOLLOW_VAR_CALL(FOLLOW_VAR,
                      printf("cpu_seq varidx %d bounds after round %d: lb: %9.2e, ub: %9.2e\n", FOLLOW_VAR, prop_round,
                             lbs[FOLLOW_VAR], ubs[FOLLOW_VAR]));

      P_inf_total += measureAndPrintProgress<datatype>(prop_round, n_vars, lbs, ubs, lbs_weakest, ubs_weakest, lbs_limit, ubs_limit,
              lbs_prev, ubs_prev, P_max, P_inf_total);
   }

   VERBOSE_CALL(printf("cpu_seq propagation done. Num rounds: %d\n", prop_round - 1));
   VERBOSE_CALL(measureTime("cpu_seq", start, std::chrono::steady_clock::now()));
   VERBOSE_CALL(printf("====   end cpu_seq with measure  ====\n"));

   VERBOSE_CALL(printf("\n====   Running the cpu_seq without measure  ====\n"));
   sequentialPropagate<datatype>(n_cons, n_vars, nnz, col_indices, row_indices, vals, lhss, rhss, lbs_orig, ubs_orig,
                                 vartypes);
   VERBOSE_CALL(printf("====   end cpu_seq without measure  ====\n"));

   free(minacts);
   free(maxacts);
   free(minacts_inf);
   free(maxacts_inf);
   free(maxactdeltas);
   free(consmarked);
   free(lbs_weakest);
   free(ubs_weakest);
   free(lbs_limit);
   free(ubs_limit);
   free(lbs_prev);
   free(ubs_prev);
   free(lbs_orig);
   free(ubs_orig);
   free(csc_vals);
   free(csc_col_ptrs);
   free(csc_row_indices);
   free(consmarked_nextround);

   return GDP_OKAY;
}


template<typename datatype>
GDP_Retcode propagateConstraintsGPUAtomicWithMeasure(
        const int n_cons,
        const int n_vars,
        const int nnz,
        const int *csr_col_indices,
        const int *csr_row_ptrs,
        const datatype *csr_vals,
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

   // need csc format of A.
   datatype *csc_vals = (datatype *) SAFEMALLOC(nnz * sizeof(datatype));
   int *csc_row_indices = (int *) SAFEMALLOC(nnz * sizeof(int));
   int *csc_col_ptrs = (int *) SAFEMALLOC((n_vars + 1) * sizeof(int));

   csr_to_csc(n_cons, n_vars, nnz, csr_col_indices, csr_row_ptrs, csc_col_ptrs, csc_row_indices, csc_vals, csr_vals);

   // keep a reference to original bounds. Will need this later.
   datatype *lbs_orig = (datatype *) SAFEMALLOC(n_vars * sizeof(datatype));
   datatype *ubs_orig = (datatype *) SAFEMALLOC(n_vars * sizeof(datatype));
   memcpy(lbs_orig, lbs, n_vars * sizeof(datatype));
   memcpy(ubs_orig, ubs, n_vars * sizeof(datatype));

   printf("\n=== gpu_atomic execution with measure of progress ===\n");
   DEBUG_CALL(checkInput<datatype>(n_cons, n_vars, nnz, csr_vals, lhss, rhss, lbs, ubs, vartypes));

   datatype *lbs_weakest = (datatype *) SAFEMALLOC(n_vars * sizeof(datatype));
   datatype *ubs_weakest = (datatype *) SAFEMALLOC(n_vars * sizeof(datatype));
   datatype *lbs_limit = (datatype *) SAFEMALLOC(n_vars * sizeof(datatype));
   datatype *ubs_limit = (datatype *) SAFEMALLOC(n_vars * sizeof(datatype));
   datatype *lbs_prev = (datatype *) malloc(n_vars * sizeof(datatype));
   datatype *ubs_prev = (datatype *) malloc(n_vars * sizeof(datatype));

   initMeasureData<datatype>
           (n_cons, n_vars, nnz, lbs, ubs, csr_col_indices, csr_row_ptrs, csr_vals,
            lhss, rhss, vartypes, csc_row_indices, csc_col_ptrs, csc_vals, lbs_weakest, ubs_weakest, lbs_limit, ubs_limit
           );

   // run atomic propagator with measure
   GPUInterface gpu = GPUInterface();

   int *d_col_indices = gpu.initArrayGPU<int>(csr_col_indices, nnz);
   int *d_row_ptrs = gpu.initArrayGPU<int>(csr_row_ptrs, n_cons + 1);
   datatype *d_vals = gpu.initArrayGPU<datatype>(csr_vals, nnz);
   datatype *d_lhss = gpu.initArrayGPU<datatype>(lhss, n_cons);
   datatype *d_rhss = gpu.initArrayGPU<datatype>(rhss, n_cons);
   datatype *d_lbs = gpu.initArrayGPU<datatype>(lbs, n_vars);
   datatype *d_ubs = gpu.initArrayGPU<datatype>(ubs, n_vars);
   GDP_VARTYPE *d_vartypes = gpu.initArrayGPU<GDP_VARTYPE>(vartypes, n_vars);

   const int blocks_count = fill_row_blocks(false, n_cons, csr_row_ptrs, nullptr);
   std::unique_ptr<int[]> row_blocks(new int[blocks_count + 1]);
   fill_row_blocks(true, n_cons, csr_row_ptrs, row_blocks.get());
   const int max_num_cons_in_block = maxConsecutiveElemDiff<int>(row_blocks.get(), blocks_count + 1);
   int *d_row_blocks = gpu.initArrayGPU<int>(row_blocks.get(), blocks_count + 1);

   datatype *d_lbs_start = gpu.initArrayGPU<datatype>(lbs_weakest, n_vars);
   datatype *d_ubs_start = gpu.initArrayGPU<datatype>(ubs_weakest, n_vars);
   datatype *d_lbs_limit = gpu.initArrayGPU<datatype>(lbs_limit, n_vars);
   datatype *d_ubs_limit = gpu.initArrayGPU<datatype>(ubs_limit, n_vars);

   const ProgressMeasure<datatype> P_max = calcMaxMeasureValues(n_vars, lbs_orig, ubs_orig, lbs_weakest, ubs_weakest, lbs_limit, ubs_limit); // maximum attainable measure
   int P_inf_total = 0;

   VERBOSE_CALL(printf("\n====   Running the gpu_atomic with measure  ====\n"));
#ifdef VERBOSE
   auto start = std::chrono::steady_clock::now();
#endif

   bool *d_change_found = gpu.allocArrayGPU<bool>(1);
   gpu.setMemGPU<bool>(d_change_found, true);
   int prop_round;
   bool change_found = true;
   for (prop_round = 1; prop_round <= MAX_NUM_ROUNDS && change_found; prop_round++) {

      gpu.setMemGPU<bool>(d_change_found, false);
      gpu.getMemFromGPU<datatype>(d_lbs, lbs_prev, n_vars);
      gpu.getMemFromGPU<datatype>(d_ubs, ubs_prev, n_vars);
      cudaDeviceSynchronize();

      // shared memory layout:
      // - max_num_cons_in_block elems of type datatype for minactivities
      // - max_num_cons_in_block elems of type datatype for maxactivities
      // - max_num_cons_in_block elems of type int for minactivities inf contributions
      // - max_num_cons_in_block elems of type int for maxactivities inf contributions
      //   VERBOSE_CALL_2(printf("Amount of dynamic shared memory requested: %.2f KB\n",
      //                         (2 * max_n_cons_in_block * sizeof(datatype)) / 1024.0));
      GPUAtomicDomainPropagation<datatype> <<< blocks_count, NNZ_PER_WG, 2 * max_num_cons_in_block *
                                                                         (sizeof(datatype) + sizeof(int)) >>>
              (
                      n_cons, max_num_cons_in_block, d_col_indices, d_row_ptrs, d_row_blocks, d_vals, d_lbs, d_ubs,
                      d_vartypes,
                      d_lhss, d_rhss, d_change_found, prop_round
              );
      cudaDeviceSynchronize();
      gpu.getMemFromGPU<bool>(d_change_found, &change_found);
      gpu.getMemFromGPU<datatype>(d_ubs, ubs, n_vars);
      gpu.getMemFromGPU<datatype>(d_lbs, lbs, n_vars);
      cudaDeviceSynchronize();

      P_inf_total += measureAndPrintProgress<datatype>(prop_round, n_vars, lbs, ubs, lbs_weakest, ubs_weakest, lbs_limit, ubs_limit,
                                                       lbs_prev, ubs_prev, P_max, P_inf_total);
   }
   VERBOSE_CALL(printf("gpu_atomic propagation done. Num rounds: %d\n", prop_round - 1));
   VERBOSE_CALL(measureTime("gpu_atomic", start, std::chrono::steady_clock::now()));
   VERBOSE_CALL(printf("====   end gpu_atomic with measure  ====\n"));


   VERBOSE_CALL(printf("\n====   Running the gpu_atomic without measure  ====\n"));
   propagateConstraintsGPUAtomic<datatype>(n_cons, n_vars, nnz, csr_col_indices, csr_row_ptrs, csr_vals, lhss, rhss,
                                           lbs_orig, ubs_orig, vartypes, CPU_LOOP);
   VERBOSE_CALL(printf("====   end gpu_atomic without measure  ====\n"));

   free(lbs_weakest);
   free(ubs_weakest);
   free(lbs_limit);
   free(ubs_limit);
   free(lbs_orig);
   free(ubs_orig);
   free(csc_vals);
   free(csc_col_ptrs);
   free(csc_row_indices);

   return GDP_OKAY;
}

#endif