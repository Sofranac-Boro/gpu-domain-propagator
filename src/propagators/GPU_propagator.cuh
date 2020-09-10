#ifndef __GPUPROPAGATOR_FULLGPUPROPAGATOR_CUH__
#define __GPUPROPAGATOR_FULLGPUPROPAGATOR_CUH__

#include <cuda_profiler_api.h>

#include "../commons.cuh"
#include "../misc.h"
#include "../kernels/atomic_kernel.cuh"
#include "../kernels/bound_reduction_kernel.cuh"
#include "../kernels/util_kernels.cuh"
#include "../params.h"
#include "../GPU_interface.cuh"
#include <memory>

template <typename datatype>
int getMaxNumConsInBlock(
    const int num_blocks,
    const int* row_blocks
)
{
    int res = 0;
    for (int i=0; i<num_blocks; i++)
    {
        int num_blocks_in_cons = row_blocks[i+1] - row_blocks[i];
        if (num_blocks_in_cons > res)
        {
            res = num_blocks_in_cons;
        }
    }
    return res;
}

template <typename datatype>
void propagateConstraintsFullGPU(
  const int n_cons,
  const int n_vars,
  const int nnz,
  int* csr_col_indices,
  int* csr_row_ptrs,
  datatype* csr_vals,
  datatype* lhss,
  datatype* rhss,
  datatype* lbs,
  datatype* ubs,
  int* vartypes
)
{
    // make sure var bounds are consistent - i.e. no integer vars have decimal values.
    consistify_var_bounds(n_vars, lbs, ubs, vartypes);

   // CUDA_CALL( cudaProfilerStart() );
    GPUInterface gpu = GPUInterface();
    
    int* d_col_indices         = gpu.initArrayGPU<int>      (csr_col_indices, nnz);
    int* d_row_ptrs            = gpu.initArrayGPU<int>      (csr_row_ptrs,    n_cons + 1);
    datatype* d_vals           = gpu.initArrayGPU<datatype> (csr_vals,        nnz);
    datatype* d_lhss           = gpu.initArrayGPU<datatype> (lhss,            n_cons);
    datatype* d_rhss           = gpu.initArrayGPU<datatype> (rhss,            n_cons);
    datatype* d_lbs            = gpu.initArrayGPU<datatype> (lbs,             n_vars);
    datatype* d_ubs            = gpu.initArrayGPU<datatype> (ubs,             n_vars);
    int* d_vartypes            = gpu.initArrayGPU<int>      (vartypes,        n_vars);
    int* d_csr2csc_index_map   = gpu.allocArrayGPU<int>(nnz);

    const int blocks_count = fill_row_blocks(false, n_cons, csr_row_ptrs, nullptr);
    std::unique_ptr<int[]> row_blocks(new int[blocks_count + 1]);
    fill_row_blocks (true, n_cons, csr_row_ptrs, row_blocks.get ());

    const int max_num_cons_in_block = getMaxNumConsInBlock<int>(blocks_count, row_blocks.get());
    
    int *d_row_blocks = gpu.initArrayGPU<int>(row_blocks.get (), blocks_count + 1);

    // Activities computed, csc matrix obtained until now.
    //new bound computation
    datatype* d_newlbs     = gpu.allocArrayGPU<datatype>(nnz);
    datatype* d_newubs     = gpu.allocArrayGPU<datatype>(nnz);
    int* d_csc_col_ptrs    = gpu.allocArrayGPU<int>(n_vars+1);
    int* d_csc_row_indices = gpu.allocArrayGPU<int>(nnz);
  
    csr_to_csc_device_only<datatype>(gpu, n_cons, n_vars, nnz, d_col_indices, d_row_ptrs, d_csc_col_ptrs, d_csc_row_indices, d_csr2csc_index_map);
    InvertMap( nnz, d_csr2csc_index_map);

    bool* d_change_found = gpu.allocArrayGPU<bool>(1);
    gpu.setMemGPU<bool>(d_change_found, true);

    #ifdef VERBOSE
    auto start = std::chrono::steady_clock::now();
    #endif
    VERBOSE_CALL( printf("\ngpu_reduction execution start...") );
     
    GPUPropEntryKernel<datatype> <<<1, 1>>>
    (
        blocks_count, max_num_cons_in_block, n_cons, n_vars, max_num_cons_in_block, d_col_indices, d_row_ptrs, d_row_blocks, d_vals, d_csc_col_ptrs, d_csc_row_indices, d_lbs,
        d_ubs, d_vartypes, d_lhss, d_rhss, d_csr2csc_index_map, d_newlbs, d_newubs, d_change_found
    );
    CUDA_CALL( cudaPeekAtLastError() );
    CUDA_CALL( cudaDeviceSynchronize() );

    VERBOSE_CALL( measureTime("gpu_reduction", start, std::chrono::steady_clock::now()) );
    gpu.getMemFromGPU<datatype>(d_ubs, ubs, n_vars);
    gpu.getMemFromGPU<datatype>(d_lbs, lbs, n_vars);
  //  VERBOSE_CALL( measureTime("Total Full GPU", start, std::chrono::steady_clock::now()) );

   // CUDA_CALL( cudaProfilerStop() );
}

template <typename datatype>
void propagateConstraintsGPUAtomic(
  const int n_cons,
  const int n_vars,
  const int nnz,
  int* csr_col_indices,
  int* csr_row_ptrs,
  datatype* csr_vals,
  datatype* lhss,
  datatype* rhss,
  datatype* lbs,
  datatype* ubs,
  int* vartypes
)
{
    // make sure var bounds are consistent - i.e. no integer vars have decimal values.
    consistify_var_bounds(n_vars, lbs, ubs, vartypes);

    // CUDA_CALL( cudaProfilerStart() );
     GPUInterface gpu = GPUInterface();

     int* d_col_indices         = gpu.initArrayGPU<int>      (csr_col_indices, nnz);
     int* d_row_ptrs            = gpu.initArrayGPU<int>      (csr_row_ptrs,    n_cons + 1);
     datatype* d_vals           = gpu.initArrayGPU<datatype> (csr_vals,        nnz);
     datatype* d_lhss           = gpu.initArrayGPU<datatype> (lhss,            n_cons);
     datatype* d_rhss           = gpu.initArrayGPU<datatype> (rhss,            n_cons);
     datatype* d_lbs            = gpu.initArrayGPU<datatype> (lbs,             n_vars);
     datatype* d_ubs            = gpu.initArrayGPU<datatype> (ubs,             n_vars);
     int* d_vartypes            = gpu.initArrayGPU<int>      (vartypes,        n_vars);

     const int blocks_count = fill_row_blocks(false, n_cons, csr_row_ptrs, nullptr);
     std::unique_ptr<int[]> row_blocks(new int[blocks_count + 1]);
     fill_row_blocks (true, n_cons, csr_row_ptrs, row_blocks.get ());

     const int max_num_cons_in_block = getMaxNumConsInBlock<int>(blocks_count, row_blocks.get());

     int *d_row_blocks = gpu.initArrayGPU<int>(row_blocks.get (), blocks_count + 1);

     bool* d_change_found = gpu.allocArrayGPU<bool>(1);
     gpu.setMemGPU<bool>(d_change_found, true);

     #ifdef VERBOSE
     auto start = std::chrono::steady_clock::now();
     #endif
     VERBOSE_CALL( printf("\ngpu_atomic execution start...") );

     GPUAtomicPropEntryKernel<datatype> <<<1, 1>>>
     (
         blocks_count, max_num_cons_in_block, n_cons, n_vars, max_num_cons_in_block, d_col_indices, d_row_ptrs, d_row_blocks, d_vals, d_lbs,
         d_ubs, d_vartypes, d_lhss, d_rhss, d_change_found
     );
     CUDA_CALL( cudaPeekAtLastError() );
     CUDA_CALL( cudaDeviceSynchronize() );

     VERBOSE_CALL( measureTime("gpu_atomic", start, std::chrono::steady_clock::now()) );
     gpu.getMemFromGPU<datatype>(d_ubs, ubs, n_vars);
     gpu.getMemFromGPU<datatype>(d_lbs, lbs, n_vars);
   //  VERBOSE_CALL( measureTime("Total Atomic GPU", start, std::chrono::steady_clock::now()) );

    // CUDA_CALL( cudaProfilerStop() );
}


#endif