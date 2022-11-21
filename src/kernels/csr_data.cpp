// #ifndef __GPUPROPAGATOR_CSRDATA_CUH__
// #define __GPUPROPAGATOR_CSRDATA_CUH__
// #include "util_kernels.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <eigen3/Eigen/Sparse>
// #ifndef __GPUPROPAGATOR_SEQUENTIAL_CUH__
// #define __GPUPROPAGATOR_SEQUENTIAL_CUH__
// #include "../propagation_methods.cuh"
// #include "../misc.h"
// #include "../params.h"
// #include <cuda.h>
// /usr/local/cuda-11.7/include
// #include "atomic_kernel.cuh"
// #include "util_kernels.cu"
// #include "util_kernels.cuh"
#include "csr_data.h"
// template<class float>
/*
int interfaceFunction(
    const int m,
    const int n,
    const int nnz,
    const float *data,
    const int *col_ind,
    const int *row_ptr,
    float *vals,
    int *row_ind,
    int *col_ptr
    )
{
    // exmaple 1:
    // int m = 5;
    // int n = 5;
    // int nnz = 10;
    // const int col_ind[] = { 0, 2, 3, 2, 4, 0, 3, 4, 2, 3 };
    // const int row_ptr[] = { 0, 3, 5, 5, 8, 10 };
    // const float data[] = { 3.0, 8.0, 2.0, 9.0, 5.0, 1.0, 4.0, 6.0, 10.0, 7.0 };

    // float vals[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    // int row_ind[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    // int col_ptr[] = { 0, 0, 0, 0, 0, 0 };

    // example 2:
    // int m = 7;
    // int n = 5;
    // int nnz = 7;
    // const int col_ind[] = { 0, 2, 2, 2, 3, 4, 3 };
    // const int row_ptr[] = { 0, 2, 3, 3, 3, 6, 6, 7 };
    // const float data[] = { 8, 2, 5, 7, 1, 2, 9 };

    // float vals[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    // int row_ind[] = { 0, 0, 0, 0, 0, 0, 0 };
    // int col_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };

    // example 3:
    // int m = 5;
    // int n = 7;
    // int nnz = 7;
    // const int col_ind[] = { 0, 0, 1, 4, 4, 6, 4 };
    // const int row_ptr[] = { 0, 1, 1, 4, 6, 7 };
    // const float data[] = { 8.0, 2.0, 5.0, 7.0, 1.0, 9.0, 2.0 };

    // float vals[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    // int row_ind[] = { 0, 0, 0, 0, 0, 0, 0 };
    // int col_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };

    // int m = 11; // number of constraints
    // int n = 12; // number of variables
    // int nnz = 52; // number of non zeros in the A matrix
    
    // const float data[] = {2, 3, 2, 9, -1, -2, -3, 5, -3, -3, 9, -2, 9, -1, 2, -4, -7, 2, 5, -2, -1, 5, 4, -5, 1, -1, 1, -2, 1, -1, -2, 1, -2, 4, 2, -1, 3, -2, -1, 5, 1, 2, -6, -2, -2, 1, 1, 1, 2, -2, 2, -2}; // dim nnz

    // const int col_ind[] = {0, 7, 8, 1, 7, 8, 1, 2, 3, 1, 3, 9, 4, 8, 9, 5, 6, 9, 6, 8, 4, 6, 8, 9, 0, 1, 2, 4, 5, 7, 8, 9, 10, 11, 1, 3, 4, 5, 7, 8, 9, 10, 11, 0, 2, 3, 4, 7, 8, 9, 10, 11}; // dim nnz

    // const int row_ptr[] = {0, 3, 6, 9, 12, 15, 18, 20, 24, 34, 43, nnz};

    
    // float *vals = (float *) calloc(nnz,sizeof(float));
    // int *row_ind = (int *) calloc(nnz,sizeof(int));
    // int *col_ptr = (int *) calloc(n+1,sizeof(int));


    // example 3:
    // int m = 11;
    // int n = 12;
    // int nnz = 52;
    // const int col_ind[] = {0, 7, 8, 1, 7, 8, 1, 2, 3, 1, 3, 9, 4, 8, 9, 5, 6, 9, 6, 8, 4, 6, 8, 9, 0, 1, 2, 4, 5, 7, 8, 9, 10, 11, 1, 3, 4, 5, 7, 8, 9, 10, 11, 0, 2, 3, 4, 7, 8, 9, 10, 11};
    // const int row_ptr[] = {0, 3, 6, 9, 12, 15, 18, 20, 24, 34, 43, nnz};
    // const float data[] = {2, 3, 2, 9, -1, -2, -3, 5, -3, -3, 9, -2, 9, -1, 2, -4, -7, 2, 5, -2, -1, 5, 4, -5, 1, -1, 1, -2, 1, -1, -2, 1, -2, 4, 2, -1, 3, -2, -1, 5, 1, 2, -6, -2, -2, 1, 1, 1, 2, -2, 2, -2};


    csr_to_csc(m, n, nnz, col_ind, row_ptr, col_ptr, row_ind, vals, data);
    // printf("vals { ");
    // for (int i =0; i <nnz;i++) {
    //     printf("%f, ",*(vals+i));
    // }
    // printf(" }");

    // printf("\nrow indices { ");
    // for (int i =0; i <nnz;i++) {
    //     printf("%d, ",*(row_ind+i));
    // }
    // printf(" }");

    // printf("\ncolumn pointers { ");
    // for (int i =0; i <n+1;i++) {
    //     printf("%d, ",*(col_ptr+i));
    // }
    // printf(" }");  

    // free(vals);
    // free(row_ind);
    // free(col_ptr);

    return 0;
}
*/
using namespace Eigen;
// int main()
// {

template <typename datatype>
void sparseMatrixConverter<datatype>::convertToCSC (
    const int m,
    const int n,
    const int nnz,
    const datatype *data,
    const int *col_ind,
    const int *row_ptr,
    datatype *vals,
    int *row_ind,
    int *col_ptr
)
{

    // // exmaple 1:
    // int m = 5;
    // int n = 44;
    // int nnz = 33;
    // // const int col_ind[] = { 0, 2, 3, 2, 4, 0, 3, 4, 2, 3 };
    // const int col_ind[] = {4,9,12,17,22,23,38,3,10,16,17,20,22,24,31,36,38,1,19,22,25,28,39,42,16,25,27,31,38,41,42,7,42};
    // // const int row_ptr[] = { 0, 3, 5, 5, 8, 10 };
    // const int row_ptr[] = {0,7,17,24,31,33};
    // // const double data[] = { 3.0, 8.0, 2.0, 9.0, 5.0, 1.0, 4.0, 6.0, 10.0, 7.0 };
    // const double data[] = {47.0,32.0,40.0,33.0,26.0,39.0,37.0,31.0,33.0,33.0,36.0,41.0,33.0,39.0,39.0,36.0,25.0,36.0,27.0,34.0,32.0,35.0,40.0,35.0,37.0,30.0,39.0,29.0,31.0,39.0,29.0,34.0,39.0};
    // double *vals = (double *) calloc(nnz,sizeof(double));
    // int *row_ind = (int *) calloc(nnz,sizeof(int));
    // int *col_ptr = (int *) calloc(n+1,sizeof(int));

    // double vals[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    // int row_ind[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    // int col_ptr[] = { 0, 0, 0, 0, 0, 0 };

    Eigen::SparseMatrix<datatype> full_matrix(m,n);

    // full_matrix.coeffRef(1,1) = 10;
    // full_matrix.coeffRef(2,2) = 10;
    // full_matrix.coeffRef(3,3) = 10;
    // full_matrix.coeffRef(4,4) = 10;
    // full_matrix.coeffRef(5,5) = 10;

    for (int i = 0;i<m;i++)
    {
        for (int j = row_ptr[i];j<row_ptr[i+1];j++)
        {
            full_matrix.insert(i,col_ind[j]) = data[j];
            // full_matrix.coeffRef(i,col_ind[j]) = data[j];
            // full_matrix[i][j] = data[j];
        }
    }
    
    full_matrix.makeCompressed();

    // vals = full_matrix.valuePtr();
    // row_ind = full_matrix.innerIndexPtr();
    // col_ptr = full_matrix.outerIndexPtr();

    // std::memcpy(vals, full_matrix.valuePtr(), sizeof vals);
    // std::memcpy(row_ind, full_matrix.innerIndexPtr(), sizeof row_ind);
    // std::memcpy(col_ptr, full_matrix.outerIndexPtr(), sizeof col_ptr);


    // Eigen::Map<Eigen::SparseMatrix<double> > full_matrix(vals,row_ind,col_ptr,values,innerIndices,outerIndexPtr);
    // Eigen::Map<Eigen::SparseMatrix<double> > full_matrix(rows, cols, nnz, row_ind,col_ptr, vals);
    
    for (int i =0; i <nnz;i++) {
        // printf("%f, ",*(vals+i));
        *(vals+i) = *(full_matrix.valuePtr()+i);
        *(row_ind+i) = *(full_matrix.innerIndexPtr()+i); 
    }

    for (int i =0; i<(n+1); i++) {
        *(col_ptr+i)= *(full_matrix.outerIndexPtr()+i);
    }
    // printf(" }");

    // printf("\nrow indices {\n");
    // for (int i =0; i <nnz;i++) {
    //     printf("%d,\n",*(row_ind+i));
    // }
    // printf(" }");

    // printf("\ncolumn pointers { \n");
    // for (int i =0; i <n+1;i++) {
    //     printf("%d,\n",*(col_ptr+i));
    // }
    // printf(" }");  

    // free(vals);
    // free(row_ind);
    // free(col_ptr);

    // return 0;
}

template class sparseMatrixConverter<float>;
template class sparseMatrixConverter<double>;

// #endif 
// pybind
// Identify which variable gives this error. 
// Julia algoirthm where it runs differently from C code. 
// Convert CSR matrix to CSC matrix in Julia.
// find eigen syntax for matrix filling.
// 