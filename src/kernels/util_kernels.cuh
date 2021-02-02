#ifndef __GPUPROPAGATOR_UTILKERNELS_CUH__
#define __GPUPROPAGATOR_UTILKERNELS_CUH__

#include "../GPU_interface.cuh"
#include "../cuda_def.cuh"
#include <cusparse_v2.h>
#include <assert.h>
#include<iostream>
#include "../params.h"
#include "../misc.h"
#include <type_traits>

#define THREADS_PER_BLOCK 256

__device__ int prev_power_of_2(int n);

int fill_row_blocks(bool fill, int rows_count, const int *row_ptr, int *row_blocks);

void CSRValidxToConsidxMap
        (
                const int n_rows,
                const int nnz,
                const int *d_row_ptrs,
                int *d_validx_considx_map
        );

void InvertMap
        (
                const int len,
                int *d_map
        );

// templates

// Performs a sum reduction of the val values in each thread in the warp.
template<class T>
__device__ T warp_reduce_sum(T val) {
   for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      val += __shfl_down_sync(FULL_WARP_MASK, val, offset);
   }
   return val;
}

// Performs a MAX reduction of the val values in each thread in the warp.
template<class T>
__device__ T warp_reduce_max(T val) {
   for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      val = fmax(__shfl_down_sync(FULL_WARP_MASK, val, offset), val);
   }
   return val;
}

template<typename datatype>
__global__ void initArrayAscendingKernel
        (
                const int len,
                datatype *array
        ) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;

   // build i to considx map
   if (i < len) {
      array[i] = i;
   }
}

template<typename datatype>
void initArrayAscending
        (
                const int len,
                datatype *array
        ) {
   const int num_threads_per_block = 256;
   const int num_blocks = ceil(double(len) / num_threads_per_block);
   initArrayAscendingKernel<datatype><<< num_blocks, num_threads_per_block >>>(len, array);
   CUDA_CALL(cudaPeekAtLastError());
   CUDA_CALL(cudaDeviceSynchronize());
}

template<typename datatype>
cudaDataType_t CtoCudaDatatype()
{
   if (std::is_same<datatype, double>::value)
   {
      return CUDA_R_64F;
   }
   else if (std::is_same<datatype, float>::value)
   {
      return CUDA_R_32F;
   }
   else
   {
      throw std::runtime_error(std::string("Unsupported datatype. Cannot convert to cuda type\n"));
   }
}


template<class datatype>
void csr_to_csc(
        const int n_cons,
        const int n_vars,
        const int nnz,
        const int *csr_col_indices,
        const int *csr_row_indices,
        int *csc_col_ptrs,
        int *csc_row_indices,
        datatype *csc_vals,
        const datatype *csr_vals
) {
   GPUInterface gpu = GPUInterface();
   int *d_col_indices = gpu.initArrayGPU<int>(csr_col_indices, nnz);
   int *d_row_ptrs = gpu.initArrayGPU<int>(csr_row_indices, n_cons + 1);
   datatype *d_vals = gpu.initArrayGPU<datatype>(csr_vals, nnz);
   int *d_csc_col_ptrs = gpu.allocArrayGPU<int>(n_vars + 1);
   int *d_csc_row_indices = gpu.allocArrayGPU<int>(nnz);
   datatype *d_vals_csc_tmp = gpu.allocArrayGPU<datatype>(nnz);

   // TODO don't need NUMERIC computation any more
   cusparseHandle_t handle = NULL;
   cudaStream_t stream = NULL;
   size_t pBufferSizeInBytes = 0;
   void *pBuffer = NULL;

   /* step 1: create cusparse handle, bind a stream */
   CUDA_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
   CUSPARSE_CALL(cusparseCreate(&handle));
   CUSPARSE_CALL(cusparseSetStream(handle, stream));

   CUSPARSE_CALL( cusparseCsr2cscEx2_bufferSize
           (
                   handle, n_cons, n_vars, nnz, d_vals, d_row_ptrs, d_col_indices, d_vals_csc_tmp,
                   d_csc_col_ptrs, d_csc_row_indices, CtoCudaDatatype<datatype>(),
                   CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG2, &pBufferSizeInBytes
           ) );

   CUDA_CALL(cudaMalloc(&pBuffer, pBufferSizeInBytes));

   // TODO when CUSPARSE_ACTION_SYMBOLIC is used, the algorithm only works on indices. However, if I pass NULL or d_vals to
   // the function instead of d_vals_csc, it somehow messes up the d_vals and d_csc_row/col. Look into this.
   CUSPARSE_CALL( cusparseCsr2cscEx2
           (
                   handle, n_cons, n_vars, nnz, d_vals, d_row_ptrs, d_col_indices, d_vals_csc_tmp,
                   d_csc_col_ptrs, d_csc_row_indices, CtoCudaDatatype<datatype>(),
                   CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG2, pBuffer
           ) );

   CUDA_CALL(cudaDeviceSynchronize());

   gpu.getMemFromGPU<datatype>(d_vals_csc_tmp, csc_vals, nnz);
   gpu.getMemFromGPU<int>(d_csc_row_indices, csc_row_indices, nnz);
   gpu.getMemFromGPU<int>(d_csc_col_ptrs, csc_col_ptrs, n_vars + 1);
   CUDA_CALL(cudaDeviceSynchronize());

   CUDA_CALL(cudaFree(pBuffer));
}

template<typename datatype>
void csr_to_csc_device_only
        (
                GPUInterface &gpu,
                const int n_cons,
                const int n_vars,
                const int nnz,
                const int *d_col_indices,
                const int *d_row_indices,
                int *d_csc_col_ptrs,
                int *d_csc_row_indices,
                int *d_csr2csc_index_map
        ) {

   //// BEGIN TEMPORARY CODE ///
   // until the INT type in vals bug is solved !
   double *csr2csc_index_map_double = (double *) SAFEMALLOC(nnz * sizeof(double));
   int *csr2csc_index_map = (int *) SAFEMALLOC(nnz * sizeof(int));
   double *d_csr2csc_index_map_double = gpu.allocArrayGPU<double>(nnz);
   /// END TEMPORARY CODE ///

   cusparseHandle_t handle = NULL;
   cudaStream_t stream = NULL;
   size_t pBufferSizeInBytes = 0;
   void *pBuffer = NULL;

   /* step 1: create cusparse handle, bind a stream */
   CUDA_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
   CUSPARSE_CALL(cusparseCreate(&handle));
   CUSPARSE_CALL(cusparseSetStream(handle, stream));

   cusparseStatus_t status1;
   cusparseStatus_t status2;

   // TODO indices should be int: CUDA_R_32I, but then the function fails. Why?
   status1 = cusparseCsr2cscEx2_bufferSize
           (
                   handle, n_cons, n_vars, nnz, NULL, NULL, NULL, NULL, NULL, NULL, CUDA_R_64F,
                   CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG2, &pBufferSizeInBytes
           );

   CUDA_CALL(cudaMalloc(&pBuffer, pBufferSizeInBytes));
   double *d_csr_indexes = gpu.allocArrayGPU<double>(nnz);
   initArrayAscending<double>(nnz, d_csr_indexes);

   status2 = cusparseCsr2cscEx2
           (
                   handle, n_cons, n_vars, nnz, d_csr_indexes, d_row_indices, d_col_indices, d_csr2csc_index_map_double,
                   d_csc_col_ptrs, d_csc_row_indices, CUDA_R_64F,
                   CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG2, pBuffer
           );

   cudaDeviceSynchronize();

   //// BEGIN TEMPORARY CODE /// ``
   gpu.getMemFromGPU<double>(d_csr2csc_index_map_double, csr2csc_index_map_double, nnz);
   for (int i = 0; i < nnz; i++) {
      csr2csc_index_map[i] = (int) csr2csc_index_map_double[i];
   }

   gpu.sendMemToGPU<int>(csr2csc_index_map, d_csr2csc_index_map, nnz);
   free(csr2csc_index_map);
   free(csr2csc_index_map_double);
   /// END TEMPORARY CODE ///


   CUDA_CALL(cudaDeviceSynchronize());

   if (status1 != CUSPARSE_STATUS_SUCCESS || status2 != CUSPARSE_STATUS_SUCCESS) {
      throw "CSR to CSC conversion error on device";
   }

   CUDA_CALL(cudaFree(pBuffer));
}

// double atomicMin
__device__ __forceinline__ double atomicMin(double *address, double val) {
   unsigned long long ret = __double_as_longlong(*address);
   while (val < __longlong_as_double(ret)) {
      unsigned long long old = ret;
      if ((ret = atomicCAS((unsigned long long *) address, old, __double_as_longlong(val))) == old)
         break;
   }
   return __longlong_as_double(ret);
}

// double atomicMax
__device__ __forceinline__ double atomicMax(double *address, double val) {
   unsigned long long ret = __double_as_longlong(*address);
   while (val > __longlong_as_double(ret)) {
      unsigned long long old = ret;
      if ((ret = atomicCAS((unsigned long long *) address, old, __double_as_longlong(val))) == old)
         break;
   }
   return __longlong_as_double(ret);
}

// float atomicMin
__device__ __forceinline__ float atomicMin(float *address, float val) {
   int ret = __float_as_int(*address);
   while (val < __int_as_float(ret)) {
      int old = ret;
      if ((ret = atomicCAS((int *) address, old, __float_as_int(val))) == old)
         break;
   }
   return __int_as_float(ret);
}

// float atomicMax
__device__ __forceinline__ float atomicMax(float *address, float val) {
   int ret = __float_as_int(*address);
   while (val > __int_as_float(ret)) {
      int old = ret;
      if ((ret = atomicCAS((int *) address, old, __float_as_int(val))) == old)
         break;
   }
   return __int_as_float(ret);
}

template<typename datatype>
__host__ __device__ __forceinline__ void atomicAdd1(datatype* val)
{
#ifdef  __CUDA_ARCH__
      atomicAdd(val, 1);
#else
      __sync_fetch_and_add(val, 1);
#endif
}

template<typename datatype>
__device__ __forceinline__ datatype adjustUpperBound(datatype ub, bool is_var_cont) {
   return is_var_cont ? ub : EPSFLOOR(ub);
}

template<typename datatype>
__device__ __forceinline__ datatype adjustLowerBound(datatype lb, bool is_var_cont) {
   return is_var_cont ? lb : EPSCEIL(lb);
}

template<typename datatype>
__device__ __forceinline__ void getNewBoundCandidates
        (
                const datatype slack,
                const datatype surplus,
                const int num_minact_inf,
                const int num_maxact_inf,
                const datatype coeff,
                const datatype lb,
                const datatype ub,
                datatype *newlb,
                datatype *newub
        ) {
   assert(!EPSEQ(coeff, 0.0));
   assert(EPSGE(ub, lb));




   // do not attempt to use the above formulas if activities or cons sides are inf. It could lead to numerical difficulties and no bound change is possibly valid.
   // lower
   //bool can_tighten = ( EPSGT(coeff, 0.0) && EPSGT(surplus, -GDP_INF) || EPSLT(coeff, 0.0) && EPSLT(slack, GDP_INF) ) && EPSGT(*newlb, -GDP_INF);
   bool can_tighten = ( EPSGT(coeff, 0.0) && EPSGT(surplus, -GDP_INF) && (num_maxact_inf == 0 || (num_maxact_inf == 1 && EPSGE(ub, GDP_INF))) );
   can_tighten = can_tighten || ( EPSLT(coeff, 0.0) && EPSLT(slack, GDP_INF) && (num_minact_inf == 0 || (num_minact_inf == 1 && EPSGE(ub, GDP_INF))) );
   bool is_one_inf_case =( EPSGT(coeff, 0.0) && num_maxact_inf == 1 && EPSGE(ub, GDP_INF) ) || ( EPSLT(coeff, 0.0) && num_minact_inf == 1 && EPSGE(ub, GDP_INF) );

   *newlb = EPSGT(coeff, 0)? surplus / coeff : slack / coeff;
   *newlb = is_one_inf_case? *newlb : *newlb + ub;
   *newlb = can_tighten && EPSGT(*newlb, -GDP_INF)? *newlb : lb;

   // upper
   //can_tighten = ( EPSGT(coeff, 0.0) && EPSLT(slack, GDP_INF) || EPSLT(coeff, 0.0) && EPSGT(surplus, -GDP_INF) ) && EPSLT(*newub, GDP_INF);
   can_tighten = ( EPSGT(coeff, 0.0) && EPSLT(slack, GDP_INF) && ( num_minact_inf == 0 || (num_minact_inf == 1 && EPSLE(lb, -GDP_INF))) );
   can_tighten = can_tighten || ( EPSLT(coeff, 0.0) && EPSGT(surplus, -GDP_INF) && ( num_maxact_inf == 0 || (num_maxact_inf == 1 && EPSLE(lb, -GDP_INF))) );
   is_one_inf_case =( EPSGT(coeff, 0.0) && num_minact_inf == 1 && EPSLE(lb, -GDP_INF) ) || ( EPSLT(coeff, 0.0) && num_maxact_inf == 1 && EPSLE(lb, -GDP_INF) );

   *newub = EPSGT(coeff, 0)? slack /coeff : surplus / coeff;
   *newub = is_one_inf_case? *newub : *newub + lb;
   *newub = can_tighten && EPSLT(*newub, GDP_INF) ? *newub : ub;

   if (EPSLT(*newub, *newlb))
   {
      printf("coeff: %9.2e, slack: %9.2e, surplus: %9.2e, newlb: %9.2e, newub: %9.2e, lb: %9.2e, ub: %9.2e, num_minact_inf: %d, num_max_act_inf: %d\n",
              coeff,        slack,        surplus,        *newlb,       *newub,       lb,        ub,        num_minact_inf,     num_maxact_inf);
   }
   assert(EPSGE(*newub, *newlb));

}

template<typename datatype>
__device__ __forceinline__ void getNewBoundCandidates_no_inf
        (
                const datatype slack,
                const datatype surplus,
                const datatype coeff,
                const datatype lb,
                const datatype ub,
                datatype *newlb,
                datatype *newub
        ) {
   assert(!EPSEQ(coeff, 0.0));
   assert(EPSGE(ub, lb));

   *newlb = EPSGT(coeff, 0)? surplus : slack;
   *newub = EPSGT(coeff, 0)? slack : surplus;
   *newlb = *newlb / coeff + ub;
   *newub = *newub / coeff + lb;

   // do not attempt to use the above formulas if activities or cons sides are inf. It could lead to numerical difficulties and no bound change is possibly valid.
   //lower
   bool can_tighten = ( EPSGT(coeff, 0.0) && EPSGT(surplus, -GDP_INF) || EPSLT(coeff, 0.0) && EPSLT(slack, GDP_INF) ) && EPSGT(*newlb, -GDP_INF);
   *newlb = can_tighten ? *newlb : lb;
   // upper
   can_tighten = ( EPSGT(coeff, 0.0) && EPSLT(slack, GDP_INF) || EPSLT(coeff, 0.0) && EPSGT(surplus, -GDP_INF) ) && EPSLT(*newub, GDP_INF);
   *newub = can_tighten ? *newub : ub;

   assert(EPSGE(*newub, *newlb));

}

template<typename datatype>
__device__ void print_acts_csr_vector(int considx, datatype minact, datatype maxact) {
   if (threadIdx.x == 0) {
      printf("cons %d: minact: %.5f, maxact: %.5f\n", considx, minact, maxact);
   }
}

template<typename datatype>
__device__ void
print_acts_csr_stream(int nnz_in_block, int *validx_considx_map, int n_cons, int block_row_begin, datatype *minacts,
                      datatype *maxacts) {
   __syncthreads();

   int considx;
   int previous_idx = validx_considx_map[0];

   if (threadIdx.x == 0) {
      for (int i = 0; i < nnz_in_block; i++) {
         considx = validx_considx_map[i];
         if (considx < n_cons && considx != previous_idx) {
            printf("cons: %d: %.5f %.5f\n", considx, minacts[considx - block_row_begin],
                   maxacts[considx - block_row_begin]);
            previous_idx = considx;
         }

      }
   }

   __syncthreads();
}

// returns score value. It is equal to lb score + ub score.
template<typename datatype>
__device__ __host__ datatype calcVarRelProgressMeasure(
        const datatype lb,
        const datatype ub,
        const datatype lb_start,
        const datatype ub_start,
        const datatype lb_limit,
        const datatype ub_limit,
        const datatype lb_prevround,
        const datatype ub_prevround,
        int *inf_change_found,
        int *rel_measure_k
) {
   assert(*rel_measure_k >= 0);
   //  lb <= lb_limit and  ub_limit <= ub
   // lb_start <= lb_limit and ub_limit <= ub_start
   assert(EPSGE(lb_limit, lb));
   assert(EPSLE(ub_limit, ub));
   assert(EPSLE(lb_start, lb_limit));
   assert(EPSGE(ub_start, ub_limit));
   // lb_start <= ub_initial and lb <= ub and lb_start <= ub_limit
   assert(EPSLE(lb_start, ub_start));
   assert(EPSLE(lb, ub));
   assert(EPSLE(lb_limit, ub_limit));

   // if limit bound is finite, so should be the starting value.
   DEBUG_CALL( EPSGT(lb_limit, -GDP_INF)? assert(EPSGT(lb_start, -GDP_INF)) : assert(true) );
   DEBUG_CALL( EPSLT(ub_limit, GDP_INF) ? assert(EPSLT(ub_start, GDP_INF)) : assert(true) );

   // if start bound is inf, it means it should never be possible to get a finite value for this bound
   DEBUG_CALL( EPSLE(lb_start, -GDP_INF)? assert(EPSLE(lb, -GDP_INF)) : assert(true) );
   DEBUG_CALL( EPSGE(ub_start, GDP_INF)? assert(EPSGE(ub, GDP_INF)) : assert(true) );

   // if start bound is finite, it should never be possible to get a "worse" finite value of the bound
   DEBUG_CALL( EPSGT(lb_start, -GDP_INF) && EPSGT(lb, -GDP_INF)? assert(EPSGE(lb, lb_start)) : assert(true) );
   DEBUG_CALL( EPSLT(ub_start,GDP_INF) && EPSLT(ub, GDP_INF)? assert(EPSLE(ub, ub_start)) : assert(true) );
   // measure contribution to finite-finite progress
   datatype score = 0.0;
   datatype increment;
   // Checking that current bound is finite prevents numerical issues even though the math works in infinite space.
   // Do not try to compute case where start value is infinite. This means that limit value is also infinite, see above.
   // lb >= lb_start is reduntant here because of the assert but added for readibility as it is present in the paper.
   // if limit == start bound, no progress can be made.
   if (EPSGT(lb_start, -GDP_INF) && EPSGT(lb, -GDP_INF) && EPSGE(lb, lb_start) && !EPSEQ(lb_limit, lb_start)) {
      increment = (lb - lb_start) / (lb_limit - lb_start);
      assert(EPSLE(increment, 1.0));
      score += increment;
   }
   // Upper bound
   if (EPSLT(ub_start, GDP_INF) && EPSLT(ub, GDP_INF) && EPSLE(ub, ub_start) && !(EPSEQ(ub_start, ub_limit))) {
      increment = (ub_start - ub)/(ub_start - ub_limit);
      assert(EPSLE(increment, 1.0));
      score += increment;
   }

   // measure contribution to infinite-finite progress
   if (EPSLE(lb_prevround, -GDP_INF) && EPSGT(lb, -GDP_INF))
   {
      *inf_change_found = 1;
      atomicAdd1<int>(rel_measure_k);
   }

   if (EPSGE(ub_prevround, GDP_INF) && EPSLT(ub, GDP_INF))
   {
      *inf_change_found = 1;
      atomicAdd1<int>(rel_measure_k);
   }

   return score;
}

template<typename datatype>
__global__ void calcRelProgressMeasure(
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
        int *inf_change_found,
        int *abs_measure_k
) {
   int varidx = blockIdx.x * blockDim.x + threadIdx.x;

   if (varidx < n_vars) {
      measures[varidx] = calcVarRelProgressMeasure(lbs[varidx], ubs[varidx], lbs_start[varidx], ubs_start[varidx],
                                                   lbs_limit[varidx], ubs_limit[varidx], lbs_prev[varidx], ubs_prev[varidx],
                                                   inf_change_found, abs_measure_k);
   }
}

template<typename datatype>
__global__ void copy_bounds_kernel(
        const int n_vars,
        datatype *lbs,
        datatype *ubs,
        const datatype *newlbs,
        const datatype *newubs
) {
   int varidx = blockIdx.x * blockDim.x + threadIdx.x;

   if (varidx < n_vars) {
      lbs[varidx] = newlbs[varidx];
      ubs[varidx] = newubs[varidx];
   }
}

template<typename datatype>
__device__ __forceinline__ bool is_change_found(
        const datatype oldlb,
        const datatype oldub,
        const datatype newlb,
        const datatype newub
) {
   assert( EPSGE(oldub, oldlb) );
   assert( EPSGE(newub, newlb) );

   return EPSLT(newub, oldub) || EPSGT(newlb, oldlb);
}

#endif