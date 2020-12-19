#ifndef __GPUPROPAGATOR_FULLGPUPROPAGATOR_CUH__
#define __GPUPROPAGATOR_FULLGPUPROPAGATOR_CUH__

#include <cuda_profiler_api.h>

#include "../def.h"
#include "../cuda_def.cuh"
#include "../misc.h"
#include "../kernels/atomic_kernel.cuh"
#include "../kernels/bound_reduction_kernel.cuh"
#include "../kernels/util_kernels.cuh"
#include "../params.h"
#include "../GPU_interface.cuh"
#include <memory>
#include <algorithm>

template<typename datatype>
void propagateConstraintsFullGPU(
        const int n_cons,
        const int n_vars,
        const int nnz,
        int *csr_col_indices,
        int *csr_row_ptrs,
        datatype *csr_vals,
        datatype *lhss,
        datatype *rhss,
        datatype *lbs,
        datatype *ubs,
        GDP_VARTYPE *vartypes
) {
   DEBUG_CALL(checkInput(n_cons, n_vars, nnz, csr_vals, lhss, rhss, lbs, ubs, vartypes));

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
   VERBOSE_CALL(printf("\ngpu_reduction execution start... Params: MAXNUMROUNDS: %d\n", MAX_NUM_ROUNDS));

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
}

template<typename datatype>
void propagateConstraintsGPUAtomic(
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
   DEBUG_CALL(checkInput(n_cons, n_vars, nnz, csr_vals, lhss, rhss, lbs, ubs, vartypes));

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

   bool change_found[2] = {true, true};
   bool *d_change_found = gpu.initArrayGPU<bool>(change_found, 2);
//gpu.setMemGPU<bool>(d_change_found, true);

#ifdef VERBOSE
   auto start = std::chrono::steady_clock::now();
#endif
   VERBOSE_CALL(printf("\ngpu_atomic execution start... Params: MAXNUMROUNDS: %d\n", MAX_NUM_ROUNDS));


// ======= NEW STUFF ==========
   //cudaDeviceProp prop;
   //cudaGetDeviceProperties(&prop, 0); // device 0. TODO make sure this is the one that will execute
   //int max_num_resident_blocks = prop.multiProcessorCount * CUDAgetMaxNumResidentBlocksPerSM(prop.major, prop.minor);
   //VERBOSE_CALL_2( printf("max num resident blocks: %d, blocks_count: %d\n", max_num_resident_blocks, blocks_count) );


   /// This will launch a grid that can maximally fill the GPU, on the default stream with kernel arguments
   int numBlocksPerSm = 0;
   // Number of threads my_kernel will be launched with
   cudaDeviceProp deviceProp;
   cudaGetDeviceProperties(&deviceProp, 0);

   // make sure the GPU supports cooperative group launch functionality
   assert(deviceProp.cooperativeLaunch == 1);

   cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, GPUAtomicDomainPropagation<datatype>, NNZ_PER_WG, 0);
   int max_num_resident_blocks = deviceProp.multiProcessorCount * numBlocksPerSm;
   printf("max num resident blocks: %d, blocks_count: %d\n", max_num_resident_blocks, blocks_count);
   // launch
   int round = 1;
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
           (void*)&round
   };
   dim3 dimBlock(NNZ_PER_WG, 1, 1);
   dim3 dimGrid(min(max_num_resident_blocks, blocks_count), 1, 1);
   cudaLaunchCooperativeKernel((void*)GPUAtomicDomainPropagation<datatype>, dimGrid, dimBlock, kernelArgs, (size_t)(2 * max_num_cons_in_block * sizeof(datatype)));



   //GPUAtomicPropEntryKernel<datatype> <<<1, 1>>>
  //         (
  //                 max_num_resident_blocks, blocks_count, n_cons, n_vars, max_num_cons_in_block, d_col_indices, d_row_ptrs, d_row_blocks, d_vals,
  //                 d_lbs, d_ubs, d_vartypes, d_lhss, d_rhss, d_change_found
  //         );
   CUDA_CALL(cudaPeekAtLastError());
   CUDA_CALL(cudaDeviceSynchronize());

   VERBOSE_CALL(measureTime("gpu_atomic", start, std::chrono::steady_clock::now()));
   gpu.getMemFromGPU<datatype>(d_ubs, ubs, n_vars);
   gpu.getMemFromGPU<datatype>(d_lbs, lbs, n_vars);

   // CUDA_CALL( cudaProfilerStop() );
}

#endif