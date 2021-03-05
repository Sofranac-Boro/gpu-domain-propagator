
#include "../../src/propagators/sequential_propagator.h"
#include "../../src/kernels/bound_reduction_kernel.cuh"
#include "../../src/kernels/util_kernels.cuh"
#include "../../src/propagators/GPU_propagator.cuh"

TEST_CASE("test CSR to CSC") {
// from Savelsbergh Preprocessing and Probing Techniques for MIP problems, page 13
// https://www2.isye.gatech.edu/~ms79/software/ojoc6.pdf

   int m = 6; // number of constraints
   int n = 6; // number of variables
   int nnz = 12; // number of non zeros in the A matrix

// Initialize problem
   std::vector<double> vals{1, 3, 1, 2, 2, 1, -15, 1, -20, 1, -5, 1}; // dim nnz
   std::vector<int> col_indices{3, 4, 3, 4, 3, 4, 0, 3, 1, 4, 2, 5}; // dim nnz
   std::vector<int> row_ptr{0, 2, 4, 6, 8, 10, nnz}; // dim m+1

   vector<int> csc_col_ptrs(n + 1, 0);
   vector<int> csc_row_indices(nnz, 0);
   vector<double> csc_vals(nnz, 0);

   csr_to_csc(m, n, nnz, col_indices.data(), row_ptr.data(), csc_col_ptrs.data(), csc_row_indices.data(),
              csc_vals.data(), vals.data());

// build expected solutions
   std::vector<double> exp_vals{1, 3, 1, 2, 2, 1, -15, 1, -20, 1, -5, 1}; // dim nnz
   std::vector<int> exp_col_indices{3, 4, 3, 4, 3, 4, 0, 3, 1, 4, 2, 5}; // dim nnz
   std::vector<int> exp_row_ptr{0, 2, 4, 6, 8, 10, nnz}; // dim m+1

   std::vector<double> exp_vals_csc{-15, -20, -5, 1, 1, 2, 1, 3, 2, 1, 1, 1};
   std::vector<int> exp_col_ptr_csc{0, 1, 2, 3, 7, 11, 12}; // dim n+1
   std::vector<int> exp_row_indices_csc{3, 4, 5, 0, 1, 2, 3, 0, 1, 2, 4, 5}; // dim nnz

   compareArrays<double>(nnz, exp_vals
                                 .

                                         data(), vals

                                 .

                                         data(),

                         1e-6, "vals");
   compareArrays<int>(nnz, exp_col_indices
                              .

                                      data(), col_indices

                              .

                                      data(),

                      1e-6, "col_indices");
   compareArrays<int>(m
                      + 1, exp_row_ptr.

                              data(), row_ptr

                              .

                                      data(),

                      1e-6, "row_ptr");

   compareCSCRandomRowOrder<double>
           (
                   m, n, nnz, csc_vals
                           .

                                   data(), csc_row_indices

                           .

                                   data(), csc_col_ptrs

                           .

                                   data(),
                   exp_vals_csc

                           .

                                   data(), exp_row_indices_csc

                           .

                                   data(), exp_col_ptr_csc

                           .

                                   data()

           );
}

TEST_CASE("test CSR to CSC device only") {
// from Savelsbergh Preprocessing and Probing Techniques for MIP problems, page 13
// https://www2.isye.gatech.edu/~ms79/software/ojoc6.pdf

   int m = 6; // number of constraints
   int n = 6; // number of variables
   int nnz = 12; // number of non zeros in the A matrix

// Initialize problem
   std::vector<double> vals{1, 3, 1, 2, 2, 1, -15, 1, -20, 1, -5, 1}; // dim nnz
   std::vector<int> col_indices{3, 4, 3, 4, 3, 4, 0, 3, 1, 4, 2, 5}; // dim nnz
   std::vector<int> row_ptr{0, 2, 4, 6, 8, 10, nnz}; // dim m+1

   vector<int> csc_col_ptrs(n + 1, 0);
   vector<int> csc_row_indices(nnz, 0);
   vector<int> csr2csc_index_map(nnz, 0);

   GPUInterface gpu = GPUInterface();
   int *d_col_indices = gpu.initArrayGPU<int>(col_indices.data(), nnz);
   int *d_row_ptr = gpu.initArrayGPU<int>(row_ptr.data(), m + 1);
   int *d_csc_col_ptrs = gpu.allocArrayGPU<int>(n + 1);
   int *d_csc_row_indices = gpu.allocArrayGPU<int>(nnz);
   int *d_csr2csc_index_map = gpu.allocArrayGPU<int>(nnz);

   csr_to_csc_device_only<double>(gpu, m, n, nnz, d_col_indices, d_row_ptr, d_csc_col_ptrs, d_csc_row_indices,
                                  d_csr2csc_index_map
   );

   gpu.
           getMemFromGPU<int>(d_csc_col_ptrs, csc_col_ptrs
           .

                   data(), n

                           + 1);
   gpu.
           getMemFromGPU<int>(d_csc_row_indices, csc_row_indices
           .

                   data(), nnz

   );
   gpu.
           getMemFromGPU<int>(d_csr2csc_index_map, csr2csc_index_map
           .

                   data(), nnz

   );

   std::vector<int> exp_col_ptr_csc{0, 1, 2, 3, 7, 11, 12}; // dim n+1
   std::vector<int> exp_row_indices_csc{3, 4, 5, 0, 1, 2, 3, 0, 1, 4, 2, 5}; // dim nnz
   std::vector<int> exp_csr2csc_index_map{6, 8, 10, 0, 2, 4, 7, 1, 3, 9, 5, 11}; // dim nnz

// need vec<double> for comparison
   std::vector<double> csr_csc_index_map_double(csr2csc_index_map.begin(), csr2csc_index_map.end());
   std::vector<double> exp_csr2csc_index_map_double(exp_csr2csc_index_map.begin(), exp_csr2csc_index_map.end());

   compareCSCRandomRowOrder<double>
           (
                   m, n, nnz, csr_csc_index_map_double
                           .

                                   data(), csc_row_indices

                           .

                                   data(), csc_col_ptrs

                           .

                                   data(),
                   exp_csr2csc_index_map_double

                           .

                                   data(), exp_row_indices_csc

                           .

                                   data(), exp_col_ptr_csc

                           .

                                   data()

           );
}

TEST_CASE("test CSCIndexInverse Kernel")
{
   int nnz = 12;
   std::vector<int> csr2csc_index_map{6, 8, 10, 0, 2, 4, 7, 1, 3, 9, 5, 11};

   GPUInterface gpu = GPUInterface();
   int *d_csr2csc_index_map = gpu.initArrayGPU<int>(csr2csc_index_map.data(), nnz);

   InvertMap(nnz, d_csr2csc_index_map
   );

   std::vector<int> exp_csr2csc_index_map_inverse(nnz, 0);
   gpu.
           getMemFromGPU<int>(d_csr2csc_index_map, csr2csc_index_map
           .

                   data(), nnz

   );

   std::vector<int> exp_csr2csc_index_map{3, 7, 4, 8, 5, 10, 0, 6, 1, 9, 2, 11};

//printArray<double>(csr2csc_index_map.data(), nnz, "csr2csc_index_map");
   compareArrays<int>(nnz, csr2csc_index_map
           .

                   data(), exp_csr2csc_index_map

                              .

                                      data()

   );
}

TEST_CASE("test CSCIndexInverse Kernel big problem")
{
   int nnz = 300;
   std::vector<int> csr2csc_index_map;
   std::vector<int> exp_csr2csc_index_map;

   for (
           int i = 0;
           i < nnz;
           i++) {
      csr2csc_index_map.
              push_back((nnz
                         - 1) - i);
      exp_csr2csc_index_map.
              push_back((nnz
                         - 1) - i);
   }

   GPUInterface gpu = GPUInterface();
   int *d_csr2csc_index_map = gpu.initArrayGPU<int>(csr2csc_index_map.data(), nnz);

   InvertMap(nnz, d_csr2csc_index_map
   );

   gpu.
           getMemFromGPU<int>(d_csr2csc_index_map, csr2csc_index_map
           .

                   data(), nnz

   );

// printArray<int>(csr2csc_index_map.data(), nnz, "csr2csc_index_map");
   compareArrays<int>(nnz, csr2csc_index_map
           .

                   data(), exp_csr2csc_index_map

                              .

                                      data()

   );
}

TEST_CASE("test CSR validx to rowidx map kernel test 1") {
// from Savelsbergh Preprocessing and Probing Techniques for MIP problems, page 13
// https://www2.isye.gatech.edu/~ms79/software/ojoc6.pdf

   int m = 6; // number of constraints
   int nnz = 12; // number of non zeros in the A matrix

// Initialize problem
   std::vector<int> row_ptr{0, 2, 4, 6, 8, 10, nnz}; // dim m+1
   std::vector<int> exp_validx_considx_map{0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5}; // dim nnz

//declare GPU memory
   GPUInterface gpu = GPUInterface();
   int *d_row_ptr = gpu.initArrayGPU<int>(row_ptr.data(), m + 1);
   int *d_validx_considx_map = gpu.allocArrayGPU<int>(nnz);

   CSRValidxToConsidxMap(m, nnz, d_row_ptr, d_validx_considx_map
   );

   std::vector<int> validx_considx_map(nnz, 0);
   gpu.
           getMemFromGPU(d_validx_considx_map, validx_considx_map
           .

                   data(), nnz

   );
   compareArrays<int>(nnz, validx_considx_map
                              .

                                      data(), exp_validx_considx_map

                              .

                                      data(),

                      1e-6, "exp_validx_considx_map");
}

TEST_CASE("test CSR validx to rowidx map kernel test 2") {
// this tests a bigger case, where there will be multiple blokcs of threads
   int nnz_per_row = 2;
   int size = 10000;

   std::vector<int> row_ptr; // dim m+1
   std::vector<int> exp_validx_considx_map; // dim nnz


   int row_ptr_counter = 0;
   for (
           int i = 0;
           i < size;
           i++) {
      for (
              int j = 0;
              j < nnz_per_row;
              j++) {
         exp_validx_considx_map.
                 push_back(i);
      }
      row_ptr.
              push_back(row_ptr_counter);
      row_ptr_counter +=
              nnz_per_row;
   }

   int nnz = row_ptr[size] = row_ptr_counter;


//declare GPU memory
   GPUInterface gpu = GPUInterface();
   int *d_row_ptr = gpu.initArrayGPU<int>(row_ptr.data(), size + 1);
   int *d_validx_considx_map = gpu.allocArrayGPU<int>(nnz);

//   printArray<int>(row_ptr.data(), size+1,               "size                ");
//   printArray<int>(exp_validx_considx_map.data(), nnz,               "exp_validx_considx_map                ");

   CSRValidxToConsidxMap(size, nnz, d_row_ptr, d_validx_considx_map
   );

   std::vector<int> validx_considx_map(nnz, 0);
   gpu.
           getMemFromGPU(d_validx_considx_map, validx_considx_map
           .

                   data(), nnz

   );
   compareArrays<int>(nnz, validx_considx_map
                              .

                                      data(), exp_validx_considx_map

                              .

                                      data(),

                      1e-6, "exp_validx_considx_map");
}

TEST_CASE("initArrayAscending")
{
   int len = 1000;

   GPUInterface gpu = GPUInterface();
   int *d_array = gpu.allocArrayGPU<int>(len);

   initArrayAscending<int>(len, d_array
   );

   std::vector<int> array(len, 0);
   gpu.
           getMemFromGPU<int>(d_array, array
           .

                   data(), len

   );

   std::vector<int> exp_array;
   for (
           int i = 0;
           i < len;
           i++) {
      exp_array.
              push_back(i);
   }

   compareArrays<int>(len, array
           .

                   data(), exp_array

                              .

                                      data()

   );
}