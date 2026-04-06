# Assignment 1: MPI matrix multiplication.
Prerequisites: cmake, ninja, openmpi, openblas... Just add what you lack. (e.g. `sudo apt install openmpi libopenblas-dev`)
```bash
mkdir build
cd build
cmake -G Ninja ..
ninja
mpirun ./MPIMatrixMultiply_bsp [OPTIONAL:<matrix dimension of A & B>]
```
by default result = A(1000, 500) * B(500, 2000).
