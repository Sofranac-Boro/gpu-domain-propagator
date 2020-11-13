#ifndef __GPUPROPAGATOR_COMMONS_CUH__
#define __GPUPROPAGATOR_COMMONS_CUH__

#include <stdio.h>


// for calling CUDA functions
#define CUDA_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


// For calling CUSPARSE functions
#define ERR_NE(X,Y) do { if ((X) != (Y)) { \
    fprintf(stderr,"Error in %s at %s:%d\n",__func__,__FILE__,__LINE__); \
    exit(-1);}} while(0)
#define CUSPARSE_CALL(X) ERR_NE((X),CUSPARSE_STATUS_SUCCESS)

#define REALABS(x)          ( fabs(x) )
#define EPSLE(x,y,eps)    ((x)-(y) <= (eps))
#define EPSLT(x,y,eps)    ((x)-(y) < -(eps))
#define EPSGE(x,y,eps)    ((x)-(y) >= -(eps))
#define EPSGT(x,y,eps)    ((x)-(y) > (eps))
#define EPSEQ(x,y,eps)      ( REALABS((x)-(y)) <= (eps) )
#define MAX(x,y)      ((x) >= (y) ? (x) : (y))
#define MIN(x,y)      ((x) <= (y) ? (x) : (y))
#define EPSFLOOR(x,eps)   (floor((x)+(eps)))
#define EPSCEIL(x,eps)    (ceil((x)-(eps)))

#endif