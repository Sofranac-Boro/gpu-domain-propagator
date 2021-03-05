#ifndef __GPUPROPAGATOR_ATOMICKERNEL_CUH__
#define __GPUPROPAGATOR_ATOMICKERNEL_CUH__

#include "../def.h"
#include "../params.h"
#include "util_kernels.cuh"
#include "../misc.h"
#include "../../lib/cub/cub.cuh"

template<typename datatype>
__global__ void GPUAtomicDomainPropagation
        (
                const int n_cons,
                const int max_n_cons_in_block,
                const int *col_indices,
                const int *row_ptrs,
                const int *row_blocks,
                const datatype *vals,
                datatype *lbs,
                datatype *ubs,
                const GDP_VARTYPE *vartypes,
                const datatype *lhss,
                const datatype *rhss,
                bool *change_found,
                const int round
        ) {
   // Launched with NNZ_PER_WG threads per block and NUM_BLOCKS blocks

   const int block_row_begin = row_blocks[blockIdx.x];
   const int block_row_end = row_blocks[blockIdx.x + 1];
   const int nnz_in_block = row_ptrs[block_row_end] - row_ptrs[block_row_begin];
   const int block_data_begin = row_ptrs[block_row_begin];

   __shared__ datatype cache_minacts[NNZ_PER_WG];
   __shared__ datatype cache_maxacts[NNZ_PER_WG];
   __shared__ int cache_inf_minacts[NNZ_PER_WG]; // used to record and compute the number of inf contributions to min activity
   __shared__ int cache_inf_maxacts[NNZ_PER_WG]; // used to record and compute the number of inf contributions to max activity
   __shared__ int validx_considx_map[NNZ_PER_WG];

   extern __shared__ unsigned char my_shared_mem[];

   datatype *shared_mem = reinterpret_cast<datatype *>(my_shared_mem);
   datatype *minacts = shared_mem;
   datatype *maxacts = &shared_mem[max_n_cons_in_block];

   int *shared_mem_int = reinterpret_cast<int *>(&shared_mem[2 * max_n_cons_in_block]);
   int *minacts_inf = shared_mem_int;
   int *maxacts_inf = &shared_mem_int[max_n_cons_in_block];

   datatype coeff;
   datatype lb;
   datatype ub;

   if (block_row_end - block_row_begin > 1) // if the block consists of more than 1 row in the matrix, use CSR-Stream
   {

      int varidx;

      // compute local contributions to activities
      if (threadIdx.x < nnz_in_block) {
         varidx = col_indices[block_data_begin + threadIdx.x];

         coeff = vals[block_data_begin + threadIdx.x];
         lb = lbs[varidx];
         ub = ubs[varidx];

         assert(EPSGT(ub, -GDP_INF));
         assert(EPSLT(lb, GDP_INF));

         cache_minacts[threadIdx.x] = EPSGT(coeff, 0) ? coeff * lb : coeff * ub; // minactivity
         cache_maxacts[threadIdx.x] = EPSGT(coeff, 0) ? coeff * ub : coeff * lb; // maxactivity

         cache_inf_minacts[threadIdx.x] = EPSGT(coeff, 0) ? EPSLE(lb, -GDP_INF) : EPSGE(ub, GDP_INF); // minactivity
         cache_inf_maxacts[threadIdx.x] = EPSGT(coeff, 0) ? EPSGE(ub, GDP_INF) : EPSLE(lb, -GDP_INF); // maxactivity

         cache_minacts[threadIdx.x] = cache_inf_minacts[threadIdx.x] ? 0.0 : cache_minacts[threadIdx.x];
         cache_maxacts[threadIdx.x] = cache_inf_maxacts[threadIdx.x] ? 0.0 : cache_maxacts[threadIdx.x];
      }
      __syncthreads();

      // power of two less than or equal to NUM_THREADS / NUM_ROWS_IN_BLOCK
      const int threads_for_reduction = prev_power_of_2(blockDim.x / (block_row_end - block_row_begin));

      if (threads_for_reduction > 1) {
         // ID of the thread in the group of threads reducing local_row
         const int thread_in_block = threadIdx.x % threads_for_reduction;

         // which row is this thread reducing
         const int local_row = block_row_begin + threadIdx.x / threads_for_reduction;

         datatype dot_minact = 0.0;
         datatype dot_maxact = 0.0;
         int dot_minact_inf = 0;
         int dot_maxact_inf = 0;

         if (local_row < block_row_end) {
            // first element in the row this thread is reducing
            const int local_first_element = row_ptrs[local_row] - block_data_begin;
            // last element in the row this thread is reducing
            const int local_last_element = row_ptrs[local_row + 1] - block_data_begin;

            //#pragma unroll
            for (int local_element = local_first_element + thread_in_block;
                 local_element < local_last_element;
                 local_element += threads_for_reduction) {
               // The following line saves the map between the index i of the vals array and the constraint j the
               // element at index i belongs to. We need this to easily access data saved per row, such as lhs and rhs.
               validx_considx_map[local_element] = local_row;
               // todo print varidx instead of threadidx.
               FOLLOW_CONS_CALL(local_row,
                                printf("considx: %7d, blockidx: %7d, threadidx: %7d, minact summand: %9.2e, maxact summand: %9.2e\n",
                                       local_row, blockIdx.x, threadIdx.x, cache_minacts[local_element],
                                       cache_maxacts[local_element]));

               dot_minact += cache_minacts[local_element];
               dot_maxact += cache_maxacts[local_element];
               dot_minact_inf += cache_inf_minacts[local_element];
               dot_maxact_inf += cache_inf_maxacts[local_element];
            }
         }

         __syncthreads();

         cache_minacts[threadIdx.x] = dot_minact;
         cache_maxacts[threadIdx.x] = dot_maxact;
         cache_inf_minacts[threadIdx.x] = dot_minact_inf;
         cache_inf_maxacts[threadIdx.x] = dot_maxact_inf;

         /// Now each row has threads_for_reduction values in cache_minacts
         //#pragma unroll
         for (int j = threads_for_reduction / 2; j > 0; j /= 2) {
            /// Reduce for each row
            __syncthreads();

            const bool use_result = thread_in_block < j && threadIdx.x + j < NNZ_PER_WG;

            if (use_result) {
               dot_minact += cache_minacts[threadIdx.x + j];
               dot_maxact += cache_maxacts[threadIdx.x + j];
               dot_minact_inf += cache_inf_minacts[threadIdx.x + j];
               dot_maxact_inf += cache_inf_maxacts[threadIdx.x + j];
            }

            __syncthreads();

            if (use_result) {
               cache_minacts[threadIdx.x] = dot_minact;
               cache_maxacts[threadIdx.x] = dot_maxact;
               cache_inf_minacts[threadIdx.x] = dot_minact_inf;
               cache_inf_maxacts[threadIdx.x] = dot_maxact_inf;
            }
         }

         if (thread_in_block == 0 && local_row < block_row_end) {
            minacts[local_row - block_row_begin] = dot_minact;
            maxacts[local_row - block_row_begin] = dot_maxact;
            minacts_inf[local_row - block_row_begin] = dot_minact_inf;
            maxacts_inf[local_row - block_row_begin] = dot_maxact_inf;
         }
      } else {
         /// Reduce all non zeroes of row by single thread
         int local_row = block_row_begin + threadIdx.x;
         const int local_first_element = row_ptrs[local_row] - block_data_begin;
         const int local_last_element = row_ptrs[local_row + 1] - block_data_begin;

         while (local_row < block_row_end) {
            datatype dot_minact = 0.0;
            datatype dot_maxact = 0.0;
            int dot_minact_inf = 0;
            int dot_maxact_inf = 0;


            //#pragma unroll
            for (int local_element = local_first_element; local_element < local_last_element; local_element++) {
               // The following line saves the map between the index i of the vals array and the constraint j the
               // element at index i belongs to. We need this to easily access data saved per row, such as lhs and rhs.
               validx_considx_map[local_element] = local_row;

               FOLLOW_CONS_CALL(local_row,
                                printf("considx: %7d, threadidx: %7d, minact summand: %9.2e, maxact summand: %9.2e\n",
                                       local_row, threadIdx.x, cache_minacts[local_element],
                                       cache_maxacts[local_element]));

               dot_minact += cache_minacts[local_element];
               dot_maxact += cache_maxacts[local_element];
               dot_minact_inf += cache_inf_minacts[local_element];
               dot_maxact_inf += cache_inf_maxacts[local_element];
            }

            minacts[local_row - block_row_begin] = dot_minact;
            maxacts[local_row - block_row_begin] = dot_maxact;
            minacts_inf[local_row - block_row_begin] = dot_minact_inf;
            maxacts_inf[local_row - block_row_begin] = dot_maxact_inf;

            local_row += NNZ_PER_WG;
         }
      }

      // COMPUTING NEW BOUND CANDIDATES FOR CSR-SPARSE CASE
      __syncthreads();

      if (threadIdx.x < nnz_in_block) {

         // get the index of the constraint that the element threadIdx.x of the vals array belongs to
         int considx = validx_considx_map[threadIdx.x];

         datatype newlb;
         datatype newub;

         getNewBoundCandidates(
                 lhss[considx], // surplus = lhs - maxact: maxacts: this is in shared memory - each block's indexing strats from 0, hence the need for - block row begin
                 rhss[considx], // slack = rhs - minact. minacts:  this is in shared memory - each block's indexing strats from 0, hence the need for - block row begin
                 minacts[considx - block_row_begin],
                 maxacts[considx - block_row_begin],
                 minacts_inf[considx - block_row_begin],
                 maxacts_inf[considx - block_row_begin],
                 coeff,
                 lb,
                 ub,
                 &newlb,
                 &newub,
                 varidx,
                 considx
         );

         FOLLOW_VAR_CALL(varidx,
                         printf("CSR-stream cand, varidx: %5d, considx: %5d, lhs: %9.2e, rhs: %9.2e, coeff: %9.2e, minact: %9.2e, maxact: %9.2e, num_minact_inf: %d, num_maxact_inf: %d, oldlb: %9.2e, oldub: %9.2e, newlb: %9.2e, newub: %9.2e\n",
                                varidx, considx, lhss[considx], rhss[considx], coeff,
                                minacts[considx - block_row_begin], maxacts[considx - block_row_begin],
                                minacts_inf[considx - block_row_begin], maxacts_inf[considx - block_row_begin],
                                lb, ub, newlb, newub)
         );

         bool is_var_cont = vartypes[varidx] == GDP_CONTINUOUS;
         newub = adjustUpperBound(newub, is_var_cont);
         newlb = adjustLowerBound(newlb, is_var_cont);

         // Candidates that do not improve old bound here don't have to be atomically checked against this rounds's new bounds. This reduces the number of atomic operations
         // This is possible because we never update a bound that is worse, i.e. if new one is not better that then old one, it's certainly not gonna be better then improvement
         // on the old bound.
         datatype oldlb = EPSGT(newlb, lb) ? atomicMax(&lbs[varidx], newlb) : lb;
         datatype oldub = EPSLT(newub, ub) ? atomicMin(&ubs[varidx], newub) : ub;

         if (is_change_found(oldlb, oldub, newlb, newub)) {
            *change_found = true;
            FOLLOW_VAR_CALL(varidx,
                            printf("CSR-stream change found for varidx: %7d, considx: %7d. oldlb: %9.2e, oldub: %9.2e, newlb: %9.2e, newub: %9.2e\n",
                                   varidx, considx, oldlb, oldub, newlb, newub)
            );
         }
      }
   } else {

      const int block_data_end = row_ptrs[block_row_begin + 1];
      const int warp_id = threadIdx.x / WARP_SIZE;
      const int lane = threadIdx.x % WARP_SIZE;

      datatype dot_minact = 0;
      datatype dot_maxact = 0;
      int dot_minact_inf = 0;
      int dot_maxact_inf = 0;

      // if nnz_in_block<=64, than computing this row with one warp is more efficient then using all the threads in the block
      if (nnz_in_block <= VECTOR_VS_VECTORL_NNZ_THRESHOLD || NNZ_PER_WG <= WARP_SIZE) {
         /// CSR-Vector case
         if (block_row_begin < n_cons) {

            //#pragma unroll
            for (int element = block_data_begin + lane; element < block_data_end; element += WARP_SIZE) {
               const int varidx = col_indices[element];
               coeff = vals[element];
               lb = lbs[varidx];
               ub = ubs[varidx];

               assert(EPSGT(ub, -GDP_INF));
               assert(EPSLT(lb, GDP_INF));

               cache_minacts[threadIdx.x] = EPSGT(coeff, 0) ? coeff * lb : coeff * ub; // minactivity
               cache_maxacts[threadIdx.x] = EPSGT(coeff, 0) ? coeff * ub : coeff * lb; // maxactivity

               cache_inf_minacts[threadIdx.x] = EPSGT(coeff, 0) ? EPSLE(lb, -GDP_INF) : EPSGE(ub,
                                                                                              GDP_INF); // minactivity
               cache_inf_maxacts[threadIdx.x] = EPSGT(coeff, 0) ? EPSGE(ub, GDP_INF) : EPSLE(lb,
                                                                                             -GDP_INF); // maxactivity

               cache_minacts[threadIdx.x] = cache_inf_minacts[threadIdx.x] ? 0.0 : cache_minacts[threadIdx.x];
               cache_maxacts[threadIdx.x] = cache_inf_maxacts[threadIdx.x] ? 0.0 : cache_maxacts[threadIdx.x];

               dot_minact += cache_minacts[threadIdx.x];
               dot_maxact += cache_maxacts[threadIdx.x];
               dot_minact_inf += cache_inf_minacts[threadIdx.x];
               dot_maxact_inf += cache_inf_maxacts[threadIdx.x];

               FOLLOW_CONS_CALL(block_row_begin,
                                printf("considx: %7d, threadidx: %7d, minact summand: %9.2e, maxact summand: %9.2e\n",
                                       block_row_begin, threadIdx.x, dot_minact, dot_maxact));
            }
         }

         __syncthreads();

         dot_minact = warp_reduce_sum<datatype>(dot_minact);
         dot_maxact = warp_reduce_sum<datatype>(dot_maxact);
         dot_minact_inf = warp_reduce_sum<int>(dot_minact_inf);
         dot_maxact_inf = warp_reduce_sum<int>(dot_maxact_inf);

         if (lane == 0 && warp_id == 0 && block_row_begin < n_cons) {
            // this is in shared memory - each block's indexing starts from 0
            minacts[0] = dot_minact;
            maxacts[0] = dot_maxact;
            minacts_inf[0] = dot_minact_inf;
            maxacts_inf[0] = dot_maxact_inf;
         }
      }
         // If there is more than 64 nnz_in_block in the row, use all the threads in the block for the computation
      else {
         /// CSR-VectorL case
         if (block_row_begin < n_cons) {

            //#pragma unroll
            for (int element = block_data_begin + threadIdx.x; element < block_data_end; element += blockDim.x) {
               const int varidx = col_indices[element];
               coeff = vals[element];
               lb = lbs[varidx];
               ub = ubs[varidx];

               assert(EPSGT(ub, -GDP_INF));
               assert(EPSLT(lb, GDP_INF));

               cache_minacts[threadIdx.x] = EPSGT(coeff, 0) ? coeff * lb : coeff * ub; // minactivity
               cache_maxacts[threadIdx.x] = EPSGT(coeff, 0) ? coeff * ub : coeff * lb; // maxactivity

               cache_inf_minacts[threadIdx.x] = EPSGT(coeff, 0) ? EPSLE(lb, -GDP_INF) : EPSGE(ub,
                                                                                              GDP_INF); // minactivity
               cache_inf_maxacts[threadIdx.x] = EPSGT(coeff, 0) ? EPSGE(ub, GDP_INF) : EPSLE(lb,
                                                                                             -GDP_INF); // maxactivity

               cache_minacts[threadIdx.x] = cache_inf_minacts[threadIdx.x] ? 0.0 : cache_minacts[threadIdx.x];
               cache_maxacts[threadIdx.x] = cache_inf_maxacts[threadIdx.x] ? 0.0 : cache_maxacts[threadIdx.x];

               dot_minact += cache_minacts[threadIdx.x];
               dot_maxact += cache_maxacts[threadIdx.x];
               dot_minact_inf += cache_inf_minacts[threadIdx.x];
               dot_maxact_inf += cache_inf_maxacts[threadIdx.x];
            }
         }

         __syncthreads();

         dot_minact = warp_reduce_sum<datatype>(dot_minact);
         dot_maxact = warp_reduce_sum<datatype>(dot_maxact);
         dot_minact_inf = warp_reduce_sum<int>(dot_minact_inf);
         dot_maxact_inf = warp_reduce_sum<int>(dot_maxact_inf);

         if (lane == 0) {
            cache_minacts[warp_id] = dot_minact;
            cache_maxacts[warp_id] = dot_maxact;
            cache_inf_minacts[warp_id] = dot_minact_inf;
            cache_inf_maxacts[warp_id] = dot_maxact_inf;
         }
         __syncthreads();

         if (warp_id == 0) {
            dot_minact = 0.0;
            dot_maxact = 0.0;
            dot_minact_inf = 0;
            dot_maxact_inf = 0;

            //#pragma unroll
            for (int element = lane; element < blockDim.x / WARP_SIZE; element += WARP_SIZE) {
               dot_minact += cache_minacts[element];
               dot_maxact += cache_maxacts[element];
               dot_minact_inf += cache_inf_minacts[element];
               dot_maxact_inf += cache_inf_maxacts[element];
            }

            dot_minact = warp_reduce_sum<datatype>(dot_minact);
            dot_maxact = warp_reduce_sum<datatype>(dot_maxact);
            dot_minact_inf = warp_reduce_sum<int>(dot_minact_inf);
            dot_maxact_inf = warp_reduce_sum<int>(dot_maxact_inf);

            if (lane == 0 && block_row_begin < n_cons) {
               // this is in shared memory - each block's indexing strats from 0
               minacts[0] = dot_minact;
               maxacts[0] = dot_maxact;
               minacts_inf[0] = dot_minact_inf;
               maxacts_inf[0] = dot_maxact_inf;
            }
         }
      }

      __syncthreads();

      // START NEW BOUND COMPUTATION CODE

      if (block_row_begin < n_cons) {

         //#pragma unroll
         for (int element = block_data_begin + threadIdx.x; element < block_data_end; element += blockDim.x) {
            const int varidx = col_indices[element];
            datatype newlb;
            datatype newub;
            lb = lbs[varidx];
            ub = ubs[varidx];

            getNewBoundCandidates(
                    lhss[block_row_begin], // surplus = lhs - maxact: maxacts: this is in shared memory - each block's indexing strats from 0
                    rhss[block_row_begin], // slack = rhs - minact. minacts: this is in shared memory - each block's indexing strats from 0
                    minacts[0],
                    maxacts[0],
                    minacts_inf[0],
                    maxacts_inf[0],
                    vals[element],
                    lb,
                    ub,
                    &newlb,
                    &newub,
                    varidx,
                    block_row_begin
            );

            FOLLOW_VAR_CALL(
                    varidx,
                    printf("CSR-vector cand: varidx: %7d, considx: %7d, lhs: %9.2e, rhs: %9.2e, coeff: %9.2e, minact: %9.2e, maxact: %9.2e, num_minact_inf: %d, num_maxact_inf: %d, oldlb: %9.2e, oldub: %9.2e, newlb: %9.2e, newub: %9.2e\n",
                           varidx, block_row_begin, lhss[block_row_begin], rhss[block_row_begin], vals[element],
                           minacts[0], maxacts[0], minacts_inf[0], maxacts_inf[0], lbs[col_indices[element]],
                           ubs[col_indices[element]], newlb, newub)
            );

            const bool is_var_cont = vartypes[varidx] == GDP_CONTINUOUS;
            newub = adjustUpperBound(newub, is_var_cont);
            newlb = adjustLowerBound(newlb, is_var_cont);

            // Candidates that do not improve old bound here don't have to be atomically checked against this rounds's new bounds. This reduces the number of atomic operations
            // This is possible because we never update a bound that is worse, i.e. if new one is not better that then old one, it's certainly not gonna be better than improvement
            // on the old bound.
            datatype oldlb = EPSGT(newlb, lb) ? atomicMax(&lbs[varidx], newlb) : lb;
            datatype oldub = EPSLT(newub, ub) ? atomicMin(&ubs[varidx], newub) : ub;

            if (is_change_found(oldlb, oldub, newlb, newub)) {
               *change_found = true;
               FOLLOW_VAR_CALL(varidx,
                               printf("CSR-vector change found for varidx: %7d, considx: %7d. oldlb: %9.2e, oldub: %9.2e, newlb: %9.2e, newub: %9.2e, minact: %9.7e, maxact: %9.7e\n",
                                      varidx, block_row_begin, oldlb, oldub, newlb, newub, minacts[0], maxacts[0])
               );
            }

         }
      }
   }
}

template<typename datatype>
__global__ void GPUAtomicPropEntryKernel
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
                const GDP_VARTYPE *vartypes,
                const datatype *lhss,
                const datatype *rhss,
                bool *change_found
        ) {
   int prop_round;

   for (prop_round = 1; prop_round <= MAX_NUM_ROUNDS && *change_found; prop_round++) {
      VERBOSE_CALL_2(printf("\nPropagation round: %d, ", prop_round));
      *change_found = false;

      FOLLOW_VAR_CALL(FOLLOW_VAR,
                      printf("\nbounds before round: %d, varidx: %7d, lb: %9.2e, ub: %9.2e\n", prop_round, FOLLOW_VAR,
                             lbs[FOLLOW_VAR],
                             ubs[FOLLOW_VAR]));
      const int num_blocks_bound_copy = ceil(double(n_vars) / BOUND_COPY_NUM_THREADS);

      // shared memory layout:
      // - max_num_cons_in_block elems of type datatype for minactivities
      // - max_num_cons_in_block elems of type datatype for maxactivities
      // - max_num_cons_in_block elems of type int for minactivities inf contributions
      // - max_num_cons_in_block elems of type int for maxactivities inf contributions
      //   VERBOSE_CALL_2(printf("Amount of dynamic shared memory requested: %.2f KB\n",
      //                         (2 * max_n_cons_in_block * sizeof(datatype)) / 1024.0));
      GPUAtomicDomainPropagation<datatype> <<< blocks_count, NNZ_PER_WG, 2 * max_n_cons_in_block *
                                                                         (sizeof(datatype) + sizeof(int)) >>>
              (
                      n_cons, max_n_cons_in_block, col_indices, row_ptrs, row_blocks, vals, lbs, ubs, vartypes,
                      lhss, rhss, change_found, prop_round
              );
      cudaDeviceSynchronize();

      FOLLOW_VAR_CALL(FOLLOW_VAR,
                      printf("bounds after round: %d, varidx: %7d, lb: %9.2e, ub: %9.2e\n", prop_round, FOLLOW_VAR,
                             lbs[FOLLOW_VAR], ubs[FOLLOW_VAR]));
   }

   VERBOSE_CALL(printf("gpu_atomic propagation done. Num rounds: %d\n", prop_round - 1));
}

#endif
