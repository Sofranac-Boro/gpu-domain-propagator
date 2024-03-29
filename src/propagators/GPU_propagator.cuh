#ifndef __GPUPROPAGATOR_FULLGPUPROPAGATOR_CUH__
#define __GPUPROPAGATOR_FULLGPUPROPAGATOR_CUH__

#include <cuda_profiler_api.h>

#include "../def.h"
#include "../cuda_def.cuh"
#include "../misc.h"
#include "../kernels/atomic_kernel.cuh"
#include "../kernels/atomic_megakernel.cuh"
#include "../kernels/bound_reduction_kernel.cuh"
#include "../kernels/util_kernels.cuh"
#include "../params.h"
#include "../GPU_interface.cuh"
#include <memory>
#include <algorithm>


template<typename datatype>
GDP_Retcode propagateConstraintsGPUReduction(
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

   DEBUG_CALL(checkInput<datatype>(n_cons, n_vars, nnz, csr_vals, lhss, rhss, lbs, ubs, vartypes));

   // CUDA_CALL( cudaProfilerStart() );
   GPUInterface gpu = GPUInterface();

   int *d_col_indices = gpu.initArrayGPU<int>(csr_col_indices, nnz);
   int *d_row_ptrs = gpu.initArrayGPU<int>(csr_row_ptrs, n_cons + 1);
   datatype *d_vals = gpu.initArrayGPU<datatype>(csr_vals, nnz);
   datatype *d_lhss = gpu.initArrayGPU<datatype>(lhss, n_cons);
   datatype *d_rhss = gpu.initArrayGPU<datatype>(rhss, n_cons);
   datatype *d_lbs = gpu.initArrayGPU<datatype>(lbs, n_vars);
   datatype *d_ubs = gpu.initArrayGPU<datatype>(ubs, n_vars);
   GDP_VARTYPE *d_vartypes = gpu.initArrayGPU<GDP_VARTYPE>(vartypes, n_vars);
   int *d_csr2csc_index_map = gpu.allocArrayGPU<int>(nnz);

   const int blocks_count = fill_row_blocks(false, n_cons, csr_row_ptrs, nullptr);
   std::unique_ptr<int[]> row_blocks(new int[blocks_count + 1]);
   fill_row_blocks(true, n_cons, csr_row_ptrs, row_blocks.get());

   const int max_num_cons_in_block = maxConsecutiveElemDiff<int>(row_blocks.get(), blocks_count + 1);

   int *d_row_blocks = gpu.initArrayGPU<int>(row_blocks.get(), blocks_count + 1);

   // Activities computed, csc matrix obtained until now.
   //new bound computation
   datatype *d_newlbs = gpu.allocArrayGPU<datatype>(nnz);
   datatype *d_newubs = gpu.allocArrayGPU<datatype>(nnz);
   int *d_csc_col_ptrs = gpu.allocArrayGPU<int>(n_vars + 1);
   int *d_csc_row_indices = gpu.allocArrayGPU<int>(nnz);

   csr_to_csc_device_only<datatype>(gpu, n_cons, n_vars, nnz, d_col_indices, d_row_ptrs, d_csc_col_ptrs,
                                    d_csc_row_indices, d_csr2csc_index_map);
   InvertMap(nnz, d_csr2csc_index_map);

   bool *d_change_found = gpu.allocArrayGPU<bool>(1);
   gpu.setMemGPU<bool>(d_change_found, true);

#ifdef VERBOSE
   auto start = std::chrono::steady_clock::now();
#endif
   VERBOSE_CALL(printf("\ngpu_reduction execution start... Datatype: %s, MAXNUMROUNDS: %d\n",
                       getDatatypeName<datatype>(), MAX_NUM_ROUNDS));

   GPUPropEntryKernel<datatype> <<<1, 1>>>
           (
                   blocks_count, n_cons, n_vars, max_num_cons_in_block, d_col_indices, d_row_ptrs, d_row_blocks, d_vals,
                   d_csc_col_ptrs, d_csc_row_indices, d_lbs,
                   d_ubs, d_vartypes, d_lhss, d_rhss, d_csr2csc_index_map, d_newlbs, d_newubs, d_change_found
           );
   CUDA_CALL(cudaPeekAtLastError());
   CUDA_CALL(cudaDeviceSynchronize());

   VERBOSE_CALL(measureTime("gpu_reduction", start, std::chrono::steady_clock::now()));
   gpu.getMemFromGPU<datatype>(d_ubs, ubs, n_vars);
   gpu.getMemFromGPU<datatype>(d_lbs, lbs, n_vars);

   // CUDA_CALL( cudaProfilerStop() );
   return GDP_OKAY;
}

template<typename datatype>
GDP_Retcode propagateConstraintsGPUAtomic(
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
        const GDP_VARTYPE *vartypes,
        const GDP_SYNCTYPE sync_type = CPU_LOOP
) {
   if (n_cons == 0 || n_vars == 0 || nnz == 0) {
      printf("propagation of 0 size problem. Nothing to propagate.\n");
      return GDP_OKAY;
   }

   DEBUG_CALL(checkInput<datatype>(n_cons, n_vars, nnz, csr_vals, lhss, rhss, lbs, ubs, vartypes));

   // CUDA_CALL( cudaProfilerStart() );
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

   VERBOSE_CALL(printf(
           "\ngpu_atomic execution start... Datatype: %s, MAXNUMROUNDS: %d, sync type: %s\n",
           getDatatypeName<datatype>(), MAX_NUM_ROUNDS, sync_type_to_str(sync_type)
   ));
#ifdef VERBOSE
   auto start = std::chrono::steady_clock::now();
#endif

   if (sync_type == GPU_LOOP) {

      bool *d_change_found = gpu.allocArrayGPU<bool>(1);
      gpu.setMemGPU<bool>(d_change_found, true);

      GPUAtomicPropEntryKernel<datatype> <<<1, 1>>>
              (
                      blocks_count, n_cons, n_vars, max_num_cons_in_block, d_col_indices, d_row_ptrs, d_row_blocks,
                      d_vals,
                      d_lbs, d_ubs, d_vartypes, d_lhss, d_rhss, d_change_found
              );
      CUDA_CALL(cudaPeekAtLastError());
      CUDA_CALL(cudaDeviceSynchronize());
   }
   else  if (sync_type == CPU_LOOP){
      bool *d_change_found = gpu.allocArrayGPU<bool>(1);
      gpu.setMemGPU<bool>(d_change_found, true);
      int prop_round;
      bool change_found = true;
      for (prop_round = 1; prop_round <= MAX_NUM_ROUNDS && change_found; prop_round++) {

         gpu.setMemGPU<bool>(d_change_found, false);

         VERBOSE_CALL_2(printf("Propagation round: %d, ", prop_round));

#ifdef FOLLOW_VAR
         datatype lb, ub;
         gpu.getMemFromGPU<datatype>(&d_lbs[FOLLOW_VAR], &lb);
         gpu.getMemFromGPU<datatype>(&d_ubs[FOLLOW_VAR], &ub);
         printf("\nbounds before round: %d, varidx: %7d, lb: %9.2e, ub: %9.2e\n", prop_round, FOLLOW_VAR, lb, ub);
#endif

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

#ifdef FOLLOW_VAR
         gpu.getMemFromGPU<datatype>(&d_lbs[FOLLOW_VAR], &lb);
         gpu.getMemFromGPU<datatype>(&d_ubs[FOLLOW_VAR], &ub);
         printf("\nbounds after round: %d, varidx: %7d, lb: %9.2e, ub: %9.2e\n", prop_round, FOLLOW_VAR, lb, ub);
#endif

         VERBOSE_CALL_2(measureTime("gpu_atomic", start, std::chrono::steady_clock::now()));
      }
      VERBOSE_CALL(printf("gpu_atomic propagation done. Num rounds: %d\n", prop_round - 1));
   }
   else if (sync_type == MEGAKERNEL)
   {
      bool change_found[2] = {true, true};
      bool *d_change_found = gpu.initArrayGPU<bool>(change_found, 2);
      int* d_round = gpu.allocArrayGPU<int>(1);

      int numBlocksPerSm = 0;
      // Number of threads my_kernel will be launched with
      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, 0);

      // make sure the GPU supports cooperative group launch functionality
      assert(deviceProp.cooperativeLaunch == 1);

      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, GPUAtomicDomainPropagation<datatype>, NNZ_PER_WG, 0);
      int max_num_resident_blocks = deviceProp.multiProcessorCount * numBlocksPerSm;
      //  printf("SMs: %d\n", deviceProp.multiProcessorCount);
      //  printf("max num resident blocks: %d, blocks_count: %d\n", max_num_resident_blocks, blocks_count);

      void *kernelArgs[16] = {
              (void*)&max_num_resident_blocks,
              (void*)&blocks_count,
              (void*)&n_cons,
              (void*)&n_vars,
              (void*)&max_num_cons_in_block,
              (void*)&d_col_indices,
              (void*)&d_row_ptrs,
              (void*)&d_row_blocks,
              (void*)&d_vals,
              (void*)&d_lbs,
              (void*)&d_ubs,
              (void*)&d_vartypes,
              (void*)&d_lhss,
              (void*)&d_rhss,
              (void*)&d_change_found,
              (void*)&d_round
      };
      dim3 dimBlock(NNZ_PER_WG, 1, 1);
      dim3 dimGrid(min(max_num_resident_blocks, blocks_count), 1, 1);

      cudaLaunchCooperativeKernel((void*)GPUAtomicDomainPropagationMegakernel<datatype>, dimGrid, dimBlock, kernelArgs, (size_t)(2 * max_num_cons_in_block * (sizeof(datatype) + sizeof(int))));


      CUDA_CALL(cudaPeekAtLastError());
      CUDA_CALL(cudaDeviceSynchronize());

      int round;
      gpu.getMemFromGPU<int>(d_round, &round);
      VERBOSE_CALL(printf("gpu_atomic propagation done. Num rounds: %d\n", round-1)); // todo rounds printing

   }
   else
   {
      throw std::runtime_error("Unknown sync type\n");
   }


   VERBOSE_CALL(measureTime("gpu_atomic", start, std::chrono::steady_clock::now()));
   gpu.getMemFromGPU<datatype>(d_ubs, ubs, n_vars);
   gpu.getMemFromGPU<datatype>(d_lbs, lbs, n_vars);

   // CUDA_CALL( cudaProfilerStop() );
   return GDP_OKAY;
}

#endif