#ifndef __GDP_PREGRESS_MEASURE_SEQ_CUH__
#define __GDP_PREGRESS_MEASURE_SEQ_CUH__

#include "../def.h"
#include "../propagators/sequential_propagator.h"
#include "preprocessor.h"
#include "../kernels/atomic_kernel.cuh"
#include <memory>


template<typename datatype>
datatype calcMaxMeasureValue
        (
                const int n_vars,
                const datatype* lbs_input,
                const datatype* ubs_input,
                const datatype* lbs_start,
                const datatype* ubs_start,
                const datatype* lbs_limit,
                const datatype* ubs_limit
        )
{
   datatype max_score = 0.0;
   int max_k = 0;
   for (int i=0; i<n_vars; i++)
   {
      if (EPSGT(lbs_limit[i], -GDP_INF) && !EPSEQ(lbs_limit[i], lbs_start[i]))
      {
         assert(EPSLE(lbs_start[i], lbs_limit[i]));
         max_score += 1.0;
      }

      if (EPSLT(ubs_limit[i], GDP_INF) && !EPSEQ(ubs_limit[i], ubs_start[i]))
      {
         assert(EPSGE(ubs_start[i], ubs_limit[i]));
         max_score += 1.0;
      }

      if (EPSGT(lbs_limit[i], -GDP_INF) && EPSLE(lbs_input[i], -GDP_INF))
      {
         max_k += 1;
      }
      if (EPSLT(ubs_limit[i], GDP_INF) && EPSGE(ubs_input[i], GDP_INF))
      {
         max_k += 1;
      }
   }
   VERBOSE_CALL_2(printf("\nMaximum measure: score=%.2f, k=%d\n", max_score, max_k));
   return max_score;
}

template<typename datatype>
datatype calcProgressMeasureSeq(
        const int n_vars,
        const datatype *lbs,
        const datatype *ubs,
        const datatype *lbs_start,
        const datatype *ubs_start,
        const datatype* lbs_limit,
        const datatype* ubs_limit,
        const datatype* lbs_prev,
        const datatype* ubs_prev,
        datatype *measures,
        int *abs_measure_k,
        int *abs_measure_n
) {

   assert(abs_measure_k >= 0);
   assert(abs_measure_n >= 0);

   datatype sum = 0.0;
   int *inf_change_found = (int *) SAFEMALLOC(sizeof(int));
   *inf_change_found = 0;

   for (int varidx = 0; varidx < n_vars; varidx++) {
      sum += calcVarRelProgressMeasure(lbs[varidx], ubs[varidx], lbs_start[varidx], ubs_start[varidx],
                                       lbs_limit[varidx], ubs_limit[varidx], lbs_prev[varidx], ubs_prev[varidx],
                                       inf_change_found, abs_measure_k);
   }

   if (*inf_change_found) {
      *abs_measure_n = *abs_measure_n + 1;
   }

   return sum;
}

// Normalizes the score to a value between 0 and 100
template<typename datatype>
__device__ __host__ __forceinline__ datatype normalizeScore(
        const datatype max_score,
        const datatype score
        )
{
   return EPSGT(max_score, 0.0)? (100.0 * score) / max_score : 0.0;

}

template<typename datatype>
void initMeasureData
        (
                const int n_cons,
                const int n_vars,
                const int nnz,
                const datatype* lbs,
                const datatype* ubs,
                const int* csr_col_indices,
                const int* csr_row_ptrs,
                const datatype* csr_vals,
                const datatype* lhss,
                const datatype* rhss,
                const GDP_VARTYPE* vartypes,
                const int* csc_row_indices,
                const int* csc_col_ptrs,
                datatype* lbs_start,
                datatype* ubs_start,
                datatype* lbs_limit,
                datatype* ubs_limit
        )
{
   // run the preprocessor to get initial bounds.
   memcpy(lbs_start, lbs, n_vars*sizeof(datatype));
   memcpy(ubs_start, ubs, n_vars*sizeof(datatype));

   printf("\n====   Running the preprocessor  ====");
   executePreprocessor<datatype>(n_cons, n_vars, csr_col_indices, csr_row_ptrs, csc_col_ptrs, csc_row_indices, csr_vals, lhss, rhss, lbs_start, ubs_start, vartypes);
   printf("====   end preprocessor  ====");
   // run sequnetial propagator to get limit bounds
   memcpy(lbs_limit, lbs, n_vars*sizeof(datatype));
   memcpy(ubs_limit, ubs, n_vars*sizeof(datatype));

   printf("\n\n==== propagator run for limit poitns ====");
   sequentialPropagate<datatype>(n_cons, n_vars, nnz, csr_col_indices, csr_row_ptrs, csr_vals, lhss, rhss, lbs_limit, ubs_limit, vartypes);
   printf("==== end limit points ====\n");
}

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
   datatype* lbs_orig = (datatype*)SAFEMALLOC(n_vars*sizeof(datatype));
   datatype* ubs_orig = (datatype*)SAFEMALLOC(n_vars*sizeof(datatype));
   memcpy(lbs_orig, lbs, n_vars * sizeof(datatype));
   memcpy(ubs_orig, ubs, n_vars * sizeof(datatype));

   // need csc format of A.
   datatype *csc_vals = (datatype *) SAFEMALLOC(nnz * sizeof(datatype));
   int *csc_row_indices = (int *) SAFEMALLOC(nnz * sizeof(int));
   int *csc_col_ptrs = (int *) SAFEMALLOC((n_vars + 1) * sizeof(int));

   csr_to_csc(n_cons, n_vars, nnz, col_indices, row_indices, csc_col_ptrs, csc_row_indices, csc_vals, vals);

   printf("\n=== cpu_seq execution with measure of progress ===\n");
   DEBUG_CALL(checkInput<datatype>(n_cons, n_vars, row_indices[n_cons], vals, lhss, rhss, lbs, ubs, vartypes));

   datatype* lbs_start = (datatype*)SAFEMALLOC(n_vars*sizeof(datatype));
   datatype* ubs_start = (datatype*)SAFEMALLOC(n_vars*sizeof(datatype));
   datatype* lbs_limit = (datatype*)SAFEMALLOC(n_vars*sizeof(datatype));
   datatype* ubs_limit = (datatype*)SAFEMALLOC(n_vars*sizeof(datatype));
   initMeasureData<datatype>
           (n_cons, n_vars, nnz, lbs, ubs, col_indices, row_indices, vals,
            lhss, rhss, vartypes, csc_row_indices, csc_col_ptrs, lbs_start,ubs_start,lbs_limit,ubs_limit
            );

   datatype* measures = (datatype*)malloc(n_vars * sizeof(datatype));
   datatype* score = (datatype*)malloc(sizeof(datatype));
   int* abs_measure_k = (int*)malloc(sizeof(int));
   int* abs_measure_n = (int*)malloc(sizeof(int));
   datatype* lbs_prev = (datatype*)malloc(n_vars * sizeof(datatype));
   datatype* ubs_prev = (datatype*)malloc(n_vars * sizeof(datatype));

   datatype *minacts = (datatype *) calloc(n_cons, sizeof(datatype));
   datatype *maxacts = (datatype *) calloc(n_cons, sizeof(datatype));
   datatype *maxactdeltas = (datatype *) calloc(n_cons, sizeof(datatype));
   int *consmarked = (int *) calloc(n_cons, sizeof(int));

   // all cons marked for propagation in the first round
   for (int i = 0; i < n_cons; i++)
      consmarked[i] = 1;

   *abs_measure_k = 0;
   *abs_measure_n = 0;

   const datatype max_score = calcMaxMeasureValue(n_vars, lbs, ubs, lbs_start, ubs_start, lbs_limit, ubs_limit);

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
                      printf("cpu_seq varidx: %7d bounds beofre round: %7d: lb: %9.2e, ub: %9.2e\n", FOLLOW_VAR, prop_round, lbs[FOLLOW_VAR],
                             ubs[FOLLOW_VAR]));

      change_found = sequentialPropagationRound<datatype>
              (
                      n_cons, n_vars, col_indices, row_indices, csc_col_ptrs, csc_row_indices, vals, lhss, rhss,
                      lbs, ubs, vartypes, minacts, maxacts, maxactdeltas, consmarked, RECOMPUTE_ACTS_TRUE
              );

      FOLLOW_VAR_CALL(FOLLOW_VAR,
                      printf("cpu_seq varidx %7d bounds after round %7d: lb: %9.2e, ub: %9.2e\n", FOLLOW_VAR, prop_round, lbs[FOLLOW_VAR], ubs[FOLLOW_VAR]));

      const datatype score = calcProgressMeasureSeq(n_vars, lbs, ubs, lbs_start, ubs_start, lbs_limit, ubs_limit, lbs_prev, ubs_prev, measures, abs_measure_k, abs_measure_n);
      assert(EPSLE(score, max_score));
      printf("round %d total score: %.10f, k=%d, n=%d\n", prop_round, normalizeScore(max_score, score), *abs_measure_k, *abs_measure_n);
   }

   VERBOSE_CALL(printf("cpu_seq propagation done. Num rounds: %d\n", prop_round - 1));
   VERBOSE_CALL(measureTime("cpu_seq", start, std::chrono::steady_clock::now()));
   VERBOSE_CALL(printf("====   end cpu_seq with measure  ====\n"));

   VERBOSE_CALL(printf("\n====   Running the cpu_seq without measure  ====\n"));
   sequentialPropagate<datatype>(n_cons, n_vars, nnz, col_indices, row_indices, vals, lhss, rhss, lbs_orig, ubs_orig, vartypes);
   VERBOSE_CALL(printf("====   end cpu_seq without measure  ====\n"));

   free(minacts);
   free(maxacts);
   free(maxactdeltas);
   free(consmarked);
   free( lbs_start );
   free(ubs_start);
   free(lbs_limit);
   free(ubs_limit);
   free(measures);
   free(abs_measure_k);
   free(abs_measure_n);
   free(lbs_prev);
   free(ubs_prev);
   free(lbs_orig);
   free(ubs_orig);
   free(csc_vals);
   free(csc_col_ptrs);
   free(csc_row_indices);

   return GDP_OKAY;
}

template<typename datatype>
__global__ void GPUAtomicPropEntryKernelWithMeasure
        (
                const int blocks_count,
                const int n_cons,
                const int n_vars,
                const int max_n_cons_in_block,
                const int *col_indices,
                const int *row_ptrs,
                const int *row_blocks,
                const datatype *vals,
                datatype *lbs,
                datatype *ubs,
                const datatype* lbs_start,
                const datatype* ubs_start,
                const datatype* lbs_limit,
                const datatype* ubs_limit,
                const datatype max_score,
                const GDP_VARTYPE *vartypes,
                const datatype *lhss,
                const datatype *rhss,
                bool *change_found
        ) {
   int prop_round;

   int* inf_change_found = (int*)malloc(sizeof(int));
   datatype* measures = (datatype*)malloc(n_vars * sizeof(datatype));
   datatype* score = (datatype*)malloc(sizeof(datatype));
   int* abs_measure_k = (int*)malloc(sizeof(int));
   int* abs_measure_n = (int*)malloc(sizeof(int));

   // Determine temporary device storage requirements
   void     *d_temp_storage = NULL;
   size_t   temp_storage_bytes = 0;
   cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, measures, score, n_vars);

   d_temp_storage = malloc(temp_storage_bytes);

   datatype* lbs_prev = (datatype*)malloc(n_vars * sizeof(datatype));
   datatype* ubs_prev = (datatype*)malloc(n_vars * sizeof(datatype));

   *abs_measure_k = 0;
   *abs_measure_n = 0;
   const int num_blocks_bound_copy = ceil(double(n_vars) / BOUND_COPY_NUM_THREADS);

   for (prop_round = 1; prop_round <= MAX_NUM_ROUNDS && *change_found; prop_round++) {

      *change_found = false;

      copy_bounds_kernel<datatype><<< num_blocks_bound_copy, BOUND_COPY_NUM_THREADS >>>(n_vars, lbs_prev, ubs_prev, lbs, ubs);

      FOLLOW_VAR_CALL(FOLLOW_VAR,
                      printf("round: %d, varidx: %7d, lb: %9.2e, ub: %9.2e\n", prop_round, FOLLOW_VAR, lbs[FOLLOW_VAR],
                             ubs[FOLLOW_VAR]));

      // shared memory layout:
      // - max_num_cons_in_block elems of type datatype for minactivities
      // - max_num_cons_in_block elems of type datatype for maxactivities
      //VERBOSE_CALL_2(printf("Amount of dynamic shared memory requested: %.2f KB\n",
      //                      (2 * max_n_cons_in_block * sizeof(datatype)) / 1024.0));
      GPUAtomicDomainPropagation<datatype> <<< blocks_count, NNZ_PER_WG, 2 * max_n_cons_in_block * sizeof(datatype) >>>
      (
              n_cons, max_n_cons_in_block, col_indices, row_ptrs, row_blocks, vals, lbs, ubs, vartypes,
                      lhss, rhss, change_found, prop_round
      );
      cudaDeviceSynchronize();

      *inf_change_found=0;
      calcRelProgressMeasure<datatype><<< num_blocks_bound_copy, BOUND_COPY_NUM_THREADS >>>
              ( n_vars, lbs, ubs, lbs_start, ubs_start, lbs_limit, ubs_limit, lbs_prev, ubs_prev, measures, inf_change_found, abs_measure_k);

      cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, measures, score, n_vars);

      cudaDeviceSynchronize();

      if (*inf_change_found)
      {
         *abs_measure_n = *abs_measure_n + 1;
      }
      // The max value for the score is 2 * n_vars.
      assert(EPSLE(*score, max_score));

      printf("round %d total score: %.10f, k=%d, n=%d\n", prop_round, normalizeScore(max_score, *score), *abs_measure_k, *abs_measure_n);

      FOLLOW_VAR_CALL(FOLLOW_VAR,
                      printf("bounds after round: %d, varidx: %7d, lb: %9.2e, ub: %9.2e\n", prop_round, FOLLOW_VAR,
                             lbs[FOLLOW_VAR], ubs[FOLLOW_VAR]));
   }


   free(d_temp_storage);
   free(measures);
   free(score);
   free(abs_measure_k);
   free(abs_measure_n);
   free(lbs_prev);
   free(ubs_prev);

   VERBOSE_CALL(printf("gpu_atomic propagation done. Num rounds: %d\n", prop_round - 1));
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
   datatype* lbs_orig = (datatype*)SAFEMALLOC(n_vars*sizeof(datatype));
   datatype* ubs_orig = (datatype*)SAFEMALLOC(n_vars*sizeof(datatype));
   memcpy(lbs_orig, lbs, n_vars * sizeof(datatype));
   memcpy(ubs_orig, ubs, n_vars * sizeof(datatype));

   printf("\n=== gpu_atomic execution with measure of progress ===\n");
   DEBUG_CALL(checkInput<datatype>(n_cons, n_vars, nnz, csr_vals, lhss, rhss, lbs, ubs, vartypes));

   datatype* lbs_start = (datatype*)SAFEMALLOC(n_vars*sizeof(datatype));
   datatype* ubs_start = (datatype*)SAFEMALLOC(n_vars*sizeof(datatype));
   datatype* lbs_limit = (datatype*)SAFEMALLOC(n_vars*sizeof(datatype));
   datatype* ubs_limit = (datatype*)SAFEMALLOC(n_vars*sizeof(datatype));
   initMeasureData<datatype>
           (n_cons, n_vars, nnz, lbs, ubs, csr_col_indices, csr_row_ptrs, csr_vals,
            lhss, rhss, vartypes, csc_row_indices, csc_col_ptrs, lbs_start,ubs_start,lbs_limit,ubs_limit
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

   bool *d_change_found = gpu.allocArrayGPU<bool>(1);
   gpu.setMemGPU<bool>(d_change_found, true);

   datatype *d_lbs_start = gpu.initArrayGPU<datatype>(lbs_start, n_vars);
   datatype *d_ubs_start = gpu.initArrayGPU<datatype>(ubs_start, n_vars);
   datatype *d_lbs_limit = gpu.initArrayGPU<datatype>(lbs_limit, n_vars);
   datatype *d_ubs_limit = gpu.initArrayGPU<datatype>(ubs_limit, n_vars);

   const datatype max_score = calcMaxMeasureValue(n_vars, lbs, ubs, lbs_start, ubs_start, lbs_limit, ubs_limit);

#ifdef VERBOSE
   auto start = std::chrono::steady_clock::now();
#endif
   VERBOSE_CALL(printf("\n====   Running the gpu_atomic with measure  ====\n"));

   GPUAtomicPropEntryKernelWithMeasure<datatype> <<<1, 1>>>
           (
                   blocks_count, n_cons, n_vars, max_num_cons_in_block, d_col_indices, d_row_ptrs,
                   d_row_blocks, d_vals, d_lbs, d_ubs, d_lbs_start, d_ubs_start, d_lbs_limit, d_ubs_limit, max_score,
                   d_vartypes, d_lhss, d_rhss, d_change_found
           );
   CUDA_CALL(cudaPeekAtLastError());
   CUDA_CALL(cudaDeviceSynchronize());

   VERBOSE_CALL(measureTime("gpu_atomic", start, std::chrono::steady_clock::now()));
   VERBOSE_CALL(printf("====   end gpu_atomic with measure  ====\n"));

   gpu.getMemFromGPU<datatype>(d_ubs, ubs, n_vars);
   gpu.getMemFromGPU<datatype>(d_lbs, lbs, n_vars);

   VERBOSE_CALL(printf("\n====   Running the gpu_atomic without measure  ====\n"));
   propagateConstraintsGPUAtomic<datatype>(n_cons, n_vars, nnz, csr_col_indices, csr_row_ptrs, csr_vals, lhss, rhss, lbs_orig, ubs_orig, vartypes, false);
   VERBOSE_CALL(printf("====   end gpu_atomic without measure  ====\n"));

   // CUDA_CALL( cudaProfilerStop() );
   free( lbs_start );
   free( ubs_start );
   free( lbs_limit );
   free( ubs_limit );
   free( lbs_orig );
   free( ubs_orig );
   free(csc_vals);
   free(csc_col_ptrs);
   free(csc_row_indices);

   return GDP_OKAY;
}
#endif