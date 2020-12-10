#ifndef __GPUPROPAGATOR_GPUINTERFACE_CUH__
#define __GPUPROPAGATOR_GPUINTERFACE_CUH__

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "cuda_def.cuh"

class GPUInterface {
private:
    std::vector<void *> gpuPtrs; // hold pointers to all allocated memory that will need to be freed later

    inline void freeMemGPU(void *ptr) {
       CUDA_CALL(cudaFree(ptr));
    }

public:

    template<typename T>
    void allocMemGPU(T **ptr, int size = 1) {
       CUDA_CALL(cudaMalloc((T **) ptr, sizeof(T) * size));
       gpuPtrs.push_back(*ptr); // add the pointer to the array of pointers to be freed in the end
    }

    template<typename T>
    inline void setMemGPU(T *ptr, T value, int size = 1) {
       CUDA_CALL(cudaMemset(ptr, value, size * sizeof(T)));
    }

    template<typename T>
    void sendMemToGPU(const T *src, T *dst, const int len = 1) {
       CUDA_CALL(cudaMemcpy(dst, src, sizeof(T) * len, cudaMemcpyHostToDevice));
    }

    template<typename T>
    void getMemFromGPU(T *src, T *dst, int len = 1) {
       CUDA_CALL(cudaMemcpy(dst, src, sizeof(T) * len, cudaMemcpyDeviceToHost));
    }

    template<typename T>
    void allocSendMemToGPU(T *src, T *dst, int len = 1) {
       allocMemGPU<T>(&dst, len);
       sendMemToGPU<T>(src, dst, len);
    }

    ~GPUInterface() {
       for (int i = 0; i < gpuPtrs.size(); i++)
          freeMemGPU(gpuPtrs[i]);
    }

    template<typename T>
    T *sendObjectToGPU(T *obj) {
       T *d_obj;
       allocMemGPU<T>(&d_obj);
       sendMemToGPU<T>(obj, d_obj);
       return d_obj;
    }

    template<typename T>
    T *initArrayGPU(const T *arr_ptr, const int num_elems) {
       T *d_arr_ptr;
       allocMemGPU<T>(&d_arr_ptr, num_elems);
       sendMemToGPU<T>(arr_ptr, d_arr_ptr, num_elems);
       return d_arr_ptr;
    }

    template<typename T>
    T *allocArrayGPU(int num_elems) {
       T *d_arr_ptr;
       allocMemGPU<T>(&d_arr_ptr, num_elems);
       return d_arr_ptr;
    }
};

#endif