cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(gpu-domain-propagator LANGUAGES CUDA CXX)

#python3
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})


# CUDA
find_package(CUDA 10.0 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
set(CMAKE_CXX_STANDARD 14)
#OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS} ")
endif ()

if (ARCH)
    string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_${ARCH},code=sm_${ARCH}")
else ()
    message("No CUDA architecture specified. To compile optimized code for your CUDA device please specify your device architecture through the ARCH flag.")
endif ()

if("${FASTMATH}" STREQUAL "On")
    message("Enabling fast math setting. This might cause floating point trouble!")
    string(APPEND CMAKE_CUDA_FLAGS " --use_fast_math")
endif()

# make 'Release' the default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
   # add_definitions(-DVERBOSE -DDEBUG)
    add_definitions(-DVERBOSE=1 -DDEBUG)
    string(APPEND CMAKE_CUDA_FLAGS_DEBUG " -G")
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")
    # add_definitions(-DVERBOSE -DDEBUG)
    add_definitions(-DVERBOSE=1)
    string(APPEND CMAKE_CUDA_FLAGS_DEBUG " -G")
endif()

if("${VERBOSITY}" STREQUAL 1)
    add_definitions(-DVERBOSE=1)
endif()

if("${VERBOSITY}" STREQUAL 2)
     add_definitions(-DVERBOSE=2)
endif()

if (FOLLOW_VAR)
    add_definitions(-DFOLLOW_VAR=${FOLLOW_VAR})
endif()

if (FOLLOW_CONS)
    add_definitions(-DFOLLOW_CONS=${FOLLOW_CONS})
endif()

# Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_library(GpuProp SHARED
        src/misc.cpp
        src/kernels/util_kernels.cu
        src/propagation_methods.cu
        src/interface.cu
        src/kernels/csr_data.cpp)

# build with -std=c++11. PUBLIC will make anything linking to this to also build with c++11
# position independent code, -fPIC
# -dc - separable compilation
target_compile_features(GpuProp PUBLIC cxx_std_14)
set_target_properties(GpuProp PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(GpuProp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET GpuProp PROPERTY CUDA_ARCHITECTURES ${ARCH})
target_link_libraries(GpuProp ${CUDA_cusparse_LIBRARY})
# Eigen3
target_link_libraries(GpuProp Eigen3::Eigen)

add_executable(testexec test/test_main.cu
        src/misc.cpp
        src/kernels/util_kernels.cu
        src/propagation_methods.cu
        test/test_infra.cu
        src/interface.cu
        src/kernels/csr_data.cpp)

set_property(TARGET testexec PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET testexec PROPERTY CUDA_ARCHITECTURES ${ARCH})
set_target_properties(testexec PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(testexec ${CUDA_cusparse_LIBRARY} GpuProp)
target_link_libraries(testexec Eigen3::Eigen GpuProp)

add_executable(gdp src/main.cpp src/kernels/csr_data.cpp)
target_link_libraries(gdp GpuProp ${PYTHON_LIBRARIES})
target_link_libraries(gdp GpuProp Eigen3::Eigen)





 

