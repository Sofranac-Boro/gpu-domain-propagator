#ifndef __GPUPROPAGATOR_UTILKERNELS_CUH__
#define __GPUPROPAGATOR_UTILKERNELS_CUH__

#include "../GPU_interface.cuh"
#include "../commons.cuh"
#include <cusparse_v2.h>
#include <assert.h>
#include<iostream>
#include "../params.h"

#define THREADS_PER_BLOCK 256

__device__ int prev_power_of_2 (int n);

int fill_row_blocks (bool fill,int rows_count,const int *row_ptr,int *row_blocks);

void CSRValidxToConsidxMap
(
    const int n_rows,
    const int nnz,
    const int* d_row_ptrs,
    int* d_validx_considx_map
);

void InvertMap
(
    const int len,
    int* d_map
);

// templates

// Performs a sum reduction of the val values in each thread in the warp.
template <class T>
__device__ T warp_reduce_sum(T val)
{
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        val += __shfl_down_sync(FULL_WARP_MASK, val, offset);
    }
    return val;
}

// Performs a MAX reduction of the val values in each thread in the warp.
template <class T>
__device__ T warp_reduce_max(T val)
{
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        val = fmax(__shfl_down_sync(FULL_WARP_MASK, val, offset), val);
    }
    return val;
}

template <typename datatype>
__global__ void initArrayAscendingKernel
(
    const int len,
    datatype* array
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // build i to considx map
    if (i < len)
    {
        array[i] = i;
    }
}

template<typename datatype>
void initArrayAscending
(
    const int len,
    datatype* array
)
{
    const int num_threads_per_block = 256;
    const int num_blocks = ceil(double(len) / num_threads_per_block);
    initArrayAscendingKernel<datatype><<< num_blocks, num_threads_per_block >>> (len, array);
    CUDA_CALL( cudaPeekAtLastError() );
    CUDA_CALL( cudaDeviceSynchronize() );
}



template <class datatype>
void csr_to_csc(
  GPUInterface& gpu,
  const int n_cons,
  const int n_vars,
  const int nnz,
  int* d_col_indices,
  int* d_row_indices,
  int* csc_col_ptrs,
  int* csc_row_indices,
  datatype* csc_vals,
  datatype* d_vals
)
{
  // TODO don't need NUMERIC computationa any more
    cusparseHandle_t handle = NULL;
    cudaStream_t stream = NULL;
    size_t pBufferSizeInBytes = 0;
    void *pBuffer = NULL;
 
    /* step 1: create cusparse handle, bind a stream */
    CUDA_CALL( cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) );
    CUSPARSE_CALL( cusparseCreate(&handle) );
    CUSPARSE_CALL( cusparseSetStream(handle, stream) );

    cusparseCsr2cscEx2_bufferSize
    (
        handle, n_cons, n_vars, nnz, NULL, NULL, NULL, NULL, NULL, NULL, CUDA_R_64F, 
        CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG2, &pBufferSizeInBytes
    );
    
    CUDA_CALL( cudaMalloc( &pBuffer, sizeof(char)* pBufferSizeInBytes) );

    // TODO convert datatype to cudaDataType
    int* d_csc_col_ptrs         = gpu.allocArrayGPU<int>(n_vars + 1); 
    int* d_csc_row_indices      = gpu.allocArrayGPU<int>(nnz);
    datatype* d_vals_csc_tmp    = gpu.allocArrayGPU<datatype>(nnz);

    // TODO when CUSPARSE_ACTION_SYMBOLIC is used, the algorithm only works on indices. However, if I pass NULL or d_vals to
    // the function instead of d_vals_csc, it somehow messes up the d_vals and d_csc_row/col. Look into this.
    cusparseCsr2cscEx2
    (
        handle, n_cons, n_vars, nnz, d_vals, d_row_indices, d_col_indices,  d_vals_csc_tmp,
        d_csc_col_ptrs, d_csc_row_indices, CUDA_R_64F, 
        CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG2, pBuffer
    );

    CUDA_CALL( cudaDeviceSynchronize() );

    gpu.getMemFromGPU<datatype>(d_vals_csc_tmp, csc_vals, nnz);
    gpu.getMemFromGPU<int>(d_csc_row_indices, csc_row_indices, nnz);
    gpu.getMemFromGPU<int>(d_csc_col_ptrs, csc_col_ptrs, n_vars + 1);
    CUDA_CALL( cudaDeviceSynchronize() );

    CUDA_CALL( cudaFree(pBuffer) );
}

template <typename datatype>
void csr_to_csc_device_only
(
    GPUInterface& gpu,
    const int n_cons,
    const int n_vars,
    const int nnz,
    const int* d_col_indices,
    const int* d_row_indices,
    int* d_csc_col_ptrs,
    int* d_csc_row_indices,
    int* d_csr2csc_index_map
)
{

    //// BEGIN TEMPORARY CODE /// 
    // until the INT type in vals bug is solved !
    double* csr2csc_index_map_double = (double*)malloc(nnz * sizeof(double));
    int* csr2csc_index_map = (int*)malloc(nnz* sizeof(int));
    double* d_csr2csc_index_map_double = gpu.allocArrayGPU<double>(nnz);
    /// END TEMPORARY CODE ///

    cusparseHandle_t handle = NULL;
    cudaStream_t stream = NULL;
    size_t pBufferSizeInBytes = 0;
    void *pBuffer = NULL;

    /* step 1: create cusparse handle, bind a stream */
    CUDA_CALL( cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) );
    CUSPARSE_CALL( cusparseCreate(&handle) );
    CUSPARSE_CALL( cusparseSetStream(handle, stream) );

    cusparseStatus_t status1;
    cusparseStatus_t status2;

    // TODO indices should be int: CUDA_R_32I, but then the function fails. Why? 
    status1 = cusparseCsr2cscEx2_bufferSize
    (
        handle, n_cons, n_vars, nnz, NULL, NULL, NULL, NULL, NULL, NULL, CUDA_R_64F, 
        CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG2, &pBufferSizeInBytes
    ); 

    CUDA_CALL( cudaMalloc( &pBuffer, sizeof(char)* pBufferSizeInBytes) );
    datatype* d_csr_indexes = gpu.allocArrayGPU<datatype>(nnz);
    initArrayAscending<datatype>(nnz, d_csr_indexes);

    status2 = cusparseCsr2cscEx2
    (
        handle, n_cons, n_vars, nnz, d_csr_indexes, d_row_indices, d_col_indices,  d_csr2csc_index_map_double,
        d_csc_col_ptrs, d_csc_row_indices, CUDA_R_64F, 
        CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG2, pBuffer
    );

    cudaDeviceSynchronize();

    //// BEGIN TEMPORARY CODE /// ``
    gpu.getMemFromGPU<double>(d_csr2csc_index_map_double, csr2csc_index_map_double, nnz);
    for (int i=0; i<nnz; i++)
    {
        csr2csc_index_map[i] = (int)csr2csc_index_map_double[i];
    }
        
    gpu.sendMemToGPU<int>(csr2csc_index_map, d_csr2csc_index_map, nnz);
    free(csr2csc_index_map);
    free(csr2csc_index_map_double);
    /// END TEMPORARY CODE ///


    CUDA_CALL( cudaDeviceSynchronize() );

    if (status1 != CUSPARSE_STATUS_SUCCESS || status2 != CUSPARSE_STATUS_SUCCESS )
    {
        throw "CSR to CSC conversion error on device";
    }

    CUDA_CALL( cudaFree(pBuffer) );
}

// double atomicMin
__device__ __forceinline__ double atomicMin(double *address, double val)
{
   unsigned long long ret = __double_as_longlong(*address);
   while(val < __longlong_as_double(ret))
   {
      unsigned long long old = ret;
      if((ret = atomicCAS((unsigned long long *)address, old, __double_as_longlong(val))) == old)
         break;
   }
   return __longlong_as_double(ret);
}

// double atomicMax
__device__ __forceinline__ double atomicMax(double *address, double val)
{
   unsigned long long ret = __double_as_longlong(*address);
   while(val > __longlong_as_double(ret))
   {
      unsigned long long old = ret;
      if((ret = atomicCAS((unsigned long long *)address, old, __double_as_longlong(val))) == old)
         break;
   }
   return __longlong_as_double(ret);
}

// float atomicMin
__device__ __forceinline__ float atomicMin(float *address, float val)
{
    int ret = __float_as_int(*address);
    while(val < __int_as_float(ret))
    {
        int old = ret;
        if((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}

// float atomicMax
__device__ __forceinline__ float atomicMax(float *address, float val)
{
    int ret = __float_as_int(*address);
    while(val > __int_as_float(ret))
    {
        int old = ret;
        if((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}

template <typename datatype>
__device__ __forceinline__ datatype adjustUpperBound(datatype ub, bool is_var_cont)
{
    return is_var_cont? ub : EPSFLOOR(ub, GDP_EPS);
}

template <typename datatype>
__device__ __forceinline__ datatype adjustLowerBound(datatype lb, bool is_var_cont)
{
    return is_var_cont? lb : EPSCEIL(lb, GDP_EPS);
}

template <typename datatype>
__device__ __forceinline__ void getNewBoundCandidates
(
  const datatype lhs,
  const datatype rhs,
  const datatype coeff,
  const datatype minact,
  const datatype maxact,
  const datatype lb,
  const datatype ub,
  datatype* newlb,
  datatype* newub
)
{
  datatype coeff_sign = coeff < 0? -1 : 1;

  datatype min_residual = coeff>0? minact - coeff*lb : minact - coeff*ub;
  datatype max_residual = coeff>0? maxact - coeff*ub : maxact - coeff*lb;

  datatype pos_newb = (rhs - min_residual) / coeff;
  datatype neg_newb = (lhs - max_residual) / coeff;

  pos_newb = rhs >=  GDP_INF?  coeff_sign*GDP_INF    : pos_newb;
  neg_newb = lhs <= -GDP_INF?  coeff_sign*(-GDP_INF) : neg_newb;

  pos_newb = minact <= -GDP_INF?  coeff_sign*GDP_INF    : pos_newb;
  neg_newb = maxact >=  GDP_INF?  coeff_sign*(-GDP_INF) : neg_newb;

  *newub = coeff>0? pos_newb : neg_newb;
  *newlb = coeff>0? neg_newb : pos_newb;
}

template<typename datatype>
__device__ void print_acts_csr_vector(int considx, datatype minact, datatype maxact)
{
    if (threadIdx.x == 0)
    {
      printf("cons %d: minact: %.5f, maxact: %.5f\n", considx, minact, maxact);
    }
}

template<typename datatype>
__device__ void print_acts_csr_stream(int nnz_in_block, int* validx_considx_map, int n_cons, int block_row_begin, datatype* minacts, datatype* maxacts)
{
        __syncthreads();

        int considx;
        int previous_idx = validx_considx_map[0];

        if (threadIdx.x == 0)
        {
              for (int i=0; i<nnz_in_block; i++)
              {
                  considx = validx_considx_map[i];
                  if (considx < n_cons && considx != previous_idx)
                  {
                        printf("cons: %d: %.5f %.5f\n", considx, minacts[considx - block_row_begin], maxacts[considx - block_row_begin]);
                        previous_idx = considx;
                  }

              }
        }

        __syncthreads();
}

template<typename datatype>
__global__ void calc_local_progress_measure(
        int n_vars,
        datatype* oldubs,
        datatype* oldlbs,
        datatype* newubs,
        datatype* newlbs,
        datatype* measures
)
{
    int varidx = blockIdx.x * blockDim.x + threadIdx.x;

    if (varidx < n_vars)
    {
            datatype oldlb = oldlbs[varidx];
            datatype oldub = oldubs[varidx];
            datatype newlb = newlbs[varidx];
            datatype newub = newubs[varidx];

            if (oldub >= GDP_INF && oldlb >= -GDP_INF)
            {
                measures[varidx] =  newub < GDP_INF || newlb < -GDP_INF? 1.0 : 0.0;
            }
            else
            {
                datatype rel_domain = oldub - oldlb < GDP_INF ? oldub - oldlb : MIN(REALABS(oldub), REALABS(oldlb));
                measures[varidx] = (oldub - newub + newlb - oldlb) / rel_domain;
            }

           // printf("var: %d, score: %.10f\n", varidx, measures[varidx]);
    }

}

#endif