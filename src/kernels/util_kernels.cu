#include "util_kernels.cuh"
#include "../GPU_interface.cuh"
#include "../cuda_def.cuh"

//compute power of two less than or equal to n
__device__ int prev_power_of_2(int n) {
   while (n & n - 1)
      n = n & n - 1;
   return n;
}

int fill_row_blocks
        (
                bool fill,
                int rows_count,
                const int *row_ptr,
                int *row_blocks
        ) {
   if (fill)
      row_blocks[0] = 0;

   int last_i = 0;
   int current_num_blocks = 1;
   int nnz_in_block_sum = 0;
   for (int i = 1; i <= rows_count; i++) {
      nnz_in_block_sum += row_ptr[i] - row_ptr[i - 1];

      if (nnz_in_block_sum == NNZ_PER_WG) {
         last_i = i;

         if (fill)
            row_blocks[current_num_blocks] = i;
         current_num_blocks++;
         nnz_in_block_sum = 0;
      } else if (nnz_in_block_sum > NNZ_PER_WG) {
         if (i - last_i > 1) {
            if (fill)
               row_blocks[current_num_blocks] = i - 1;
            current_num_blocks++;
            i--;
         } else {
            if (fill)
               row_blocks[current_num_blocks] = i;
            current_num_blocks++;
         }

         last_i = i;
         nnz_in_block_sum = 0;
      } else if (i - last_i > NNZ_PER_WG) {
         last_i = i;
         if (fill)
            row_blocks[current_num_blocks] = i;
         current_num_blocks++;
         nnz_in_block_sum = 0;
      }
   }

   if (fill)
      row_blocks[current_num_blocks] = rows_count;

   return current_num_blocks;
}

__global__ void CSRValidxToConsidxMapKernel
        (
                const int n_rows,
                const int nnz,
                const int *row_ptrs,
                int *validx_considx_map
        ) {
   int rowidx = blockIdx.x * blockDim.x + threadIdx.x;

   // build i to considx map
   if (rowidx < n_rows) {
      // thread i takes position block_row_begin + i until block row begin + i + 1
      int beginidx = row_ptrs[rowidx];
      int endidx = row_ptrs[rowidx + 1];
      //printf("threadidx-considx: %d, indices from: %d, to %d\n", rowidx, beginidx, endidx);
      for (int validx = beginidx; validx < endidx; validx++) {
         validx_considx_map[validx] = rowidx;
      }
   }
}

// Interfaces

void CSRValidxToConsidxMap
        (
                const int n_rows,
                const int nnz,
                const int *d_row_ptrs,
                int *d_validx_considx_map
        ) {
   //TODO make asynchronous
   const int num_threads_per_block = 256;
   const int num_blocks = ceil(double(n_rows) / num_threads_per_block);
   CSRValidxToConsidxMapKernel << < num_blocks, num_threads_per_block >> >
   (n_rows, nnz, d_row_ptrs, d_validx_considx_map);
   CUDA_CALL(cudaPeekAtLastError());
   CUDA_CALL(cudaDeviceSynchronize());
}

__global__ void InvertMapKernel
        (
                const int len,
                int *input_map,
                int *output_map
        ) {

   int i = blockIdx.x * blockDim.x + threadIdx.x;

   if (i < len) {
      int dest_idx = input_map[i];
      output_map[dest_idx] = i;

   }
}

void InvertMap
        (
                const int len,
                int *d_map
        ) {
   //allocate temporary storage
   int *d_input_map;
   CUDA_CALL(cudaMalloc(&d_input_map, sizeof(int) * len));
   CUDA_CALL(cudaMemcpy(d_input_map, d_map, sizeof(int) * len, cudaMemcpyHostToDevice));

   const int num_blocks = ceil(double(len) / THREADS_PER_BLOCK);
   InvertMapKernel << < num_blocks, THREADS_PER_BLOCK >> > (len, d_input_map, d_map);

   CUDA_CALL(cudaDeviceSynchronize());
   CUDA_CALL(cudaFree(d_input_map));
}