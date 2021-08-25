# GPU Domain Propagator

Given a Mixed Integer Linear Program (MIP), this code performs iterated domain propagation on a GPU. It can be used as a C shared library. For testing purposes a Python file reader is provided. The GPU functions could also be used to embed the GPU algorithms in a GPU only program. 
The following algorithm implementations are provided:
* **cpu_seq** is the sequential (single-threaded) implementation of domain propagation
* **cpu_omp** is the parallelized version of the algorithm above. Parallelization is done in shared CPU memory with OpenMP.
* **gpu_atomic** is a GPU implementation of domain propagation. It uses atomic updates in GPU global memory to resolve dependencies. After sending the initial input memory to the GPU, this algorithm is capable of running fully on the GPU, requiring no interaction with the CPU. A better-performing version of the algorithm with minimal CPU synchronization is also available, for the case where minimal CPU involvemnt is acceptable.

This repository contains all the methods used to generate results of [this paper](https://arxiv.org/abs/2009.07785).
This code also contains methods for measuring progress of iterative domain propagation algorithms, see [this paper](https://arxiv.org/abs/2106.07573).

## Building

### Dependencies
Core dependencies are:
* CUDA enabled GPU, CUDA >= 10.0
* CMake >= 3.8
* C++ compiler

For the usage of the shared memory algorithm:
* OpenMP

For the usage of the Python file reader:
* Python3
* pip3

Necessary Python packages are built automatically with pip3

### Compilation

To compile the code, after downloading and `cd`ing into the home folder, execute the following commands:
```
$ mkdir build
$ cd build
$ cmake .. -DARCH=<compute_capability>
$ make
```
The `compute_capability` is the compute capability of
your device, as passed to the nvcc compiler. For ex-
ample, for a compute capability 6.0, pass 60; for compute capability 7.5, pass 75. To set the number of
OpenMP threads the *cpu_omp* algorithm should use, set the
`SHARED_MEM_THREADS` macro in `src/params.h` to the
desired value before the installation process. The default value
is 8. This parameters file also contains a number of different parameters that control the internal behavior of the algorithms which the user can set to the desired values.

If you intend to use the file reader, please also install the Python dependencies. From the home folder, do:
```
$ cd fileReader/
$ pip3 install -r requirements.txt
```
Using a virtual environment is highly recommended but optional.


### Testing your installation
The compilation process creates a `testexec` executable in the `build` folder. You can run all the tests simply with:
```
$ cd build/
$ ./testexec
```
You can also execute individual tests with:
```
$ cd build/
$ ./testexec <test_name>
```
You can see all the tests available in the files in `test/testCases`

## Using the gpu-domain-propagator as a C callable library

The compilation process creates a C shared library file `libGpuProp.so` in the `build` folder. Available interface functions can be found in `include/interface.h`.

## Testing by using the Python file reader

For testing purposes, the code supports reading standard MIP file formats such as `.mps` and executing the available algorithms on it. For this, we use the `mip` package to load standard MIP file formats. For the full description of supported file formats see https://python-mip.com/.

To read a MIP file and execute all the algorithms on it, execute the following commands from the home folder:
```
$ cd fileReader/
$ ./exec_file.sh
Usage: ./exec_file.sh -f|--file <FILE> -l|--logfile <LOGFILE>
<FILE>: path to the input file in e.g. .lp or .mps format
<LOGFILE>L path to the log file to save output
```
This executes all the algorithms on the given file and prints some statistics such as execution times and whether the results of individual algorithms match.

A scirpt is also provided to execute the algorithms on all the MIP files in a given path. This helps automate testing process on larger test sets:
```
$ cd fileReader/
$ ./exec_tests.sh
Usage: ./exec_tests.sh -f|--files <FILES> -l|--logfile <LOGFILE>
<FILES>: path to the folder containing input files in e.g. .lp or .mps format
<LOGFILE>L path to the log file to save output
```
## Miscellaneous Notes

- Exact code used for the submission to the IAAA 10th Workshop on Irregular Applications: Architectures and Algorithms can be found on the `IAAA_version` branch
