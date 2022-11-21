#ifndef __GPUPROPAGATOR_CSRCSC_H__
#define __GPUPROPAGATOR_CSRCSC_H__

// #ifdef __cplusplus
// extern "C" {
// #endif
template<typename datatype>
class sparseMatrixConverter
{
    public:
        void convertToCSC(
        const int m,
        const int n,
        const int nnz,
        const datatype *data,
        const int *col_ind,
        const int *row_ptr,
        datatype *vals,
        int *row_ind,
        int *col_ptr
    );
};

// typedef convertToCSC<float> FCsrToCsc;
// typedef convertToCSC<double> DCsrToCsc;
// typedef sparseMatrixConverter<float> FCsrToCsc;
// typedef sparseMatrixConverter<double> DCsrToCsc;
// typedef sparseMatrixConverter<double> DCsrToCsc;

// typedef sparseMatrixConverter<> CsrToCsc;

    // template <class datatype>

// #ifdef __cplusplus
// }
// #endif
#endif