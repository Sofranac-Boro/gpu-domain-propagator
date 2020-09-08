cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(gpu-domain-propagator LANGUAGES CUDA CXX)

# CUDA
find_package(CUDA 10.0 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

#OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS} ")
endif()

if (ARCH)
    string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_${ARCH},code=sm_${ARCH}")
else()
    message("No CUDA architecture specified. To compile optimized code for your CUDA device please specify your device architecture through the ARCH flag.")
endif()

#string(APPEND CMAKE_CUDA_FLAGS_DEBUG " -G")


add_library(GpuProp SHARED
        src/misc.cpp
        src/kernels/util_kernels.cu
        src/propagation_methods.cpp
  src/interface.cu)

# build with -std=c++11. PUBLIC will make anything linking to this to also build with c++11
 # position independent code, -fPIC
 # -dc - separable compilation
target_compile_features(GpuProp PUBLIC cxx_std_11)
set_target_properties(GpuProp PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(GpuProp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries( GpuProp ${CUDA_cusparse_LIBRARY} )

add_executable(testexec test/test_main.cu
        src/misc.cpp
        src/kernels/util_kernels.cu
        src/propagation_methods.cpp
        test/test_infra.cu
  src/interface.cu)
set_property(TARGET testexec PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(testexec PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(testexec ${CUDA_cusparse_LIBRARY} GpuProp)



 

