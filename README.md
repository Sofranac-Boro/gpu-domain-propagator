# GPU Domain Propagator

Given a Mixed Integer Linear Program (MIP), this code performs iterated domain propagation on a GPU. It can be used either through a C shared library or as a standalone program. For testing purposes (or for usage as a standalone program), a Python file reader is provided.
The following algorithm implementations are provided:
* **cpu_seq** is the sequential (single threaded) implementation of domain propagation
* **cpu_omp** is the parallelized version of the algorithm above. Parallelization is done in shared CPU memory with OpenMP.
* **gpu_atomic** is a GPU implementation of domain propagation. It uses atomic updates in GPU global memory to resolve dependencies.
* **gpu_reduciton** is a version of the GPU implementation which avoids using atomics by saving the data in global memory followed by a reduciton.

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
mkdir build
cd build
cmake .. -DARCH=<compute_capability>
make
```
The `compute_capability` is the compute capability of
your device, as passed to the nvcc compiler. For ex-
ample, for a compute capability 6.0, pass 60; for com-
pute capability 7.5, pass 75. To set the number of
OpenMP threads the cpu omp algorithm should use, set the
`SHARED_MEM_THREADS` macro in `src/params.h` to the
desired value before the installation process. The default value
is 8.

If you intend to use the file reader, please also isntall the Python dependencies:
```
cd ../fileReader/
pip3 install -r requirements.txt
```
Using a virtual environment is highly recommended but optional.


### Testing your installation
The compilation process creates a `testexec` executable in the `build` folder. You can run all the tests simply with:
```
./testexec
```
You can also execute individual tests with:
```
./testexec <test_name>
```
You can see all the tests available in the files in `test/testCases`

## Using gpu-domain-propagator as a C callable library

The compilation process creates a C shared library file `libGpuProp.so` in the `build` folder.  
