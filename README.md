# parallel-dense-vs-csr-openmp
OpenMP-based comparison of dense and CSR matrix-vector multiplication in C.
Project summary

Parallel C implementation that compares dense and CSR (Compressed Sparse Row) matrix–vector multiplication using OpenMP. The program generates a square matrix with a configurable percentage of zeros, computes matrix–vector products using both dense and CSR representations, measures execution time, and verifies correctness.


Contents of this repository
parallel-dense-vs-csr-openmp/
├── src/
│   └── main.c        # your program
├── Makefile
├── README.md         # this file
├── .gitignore
└── LICENSE
Requirements

A C compiler with OpenMP support (e.g. gcc or clang with -fopenmp).

make utility

POSIX-compatible OS (Linux, macOS)

Tested with gcc + OpenMP and mpicc is not required for this OpenMP version.

Usage
./main <n> <pct_zeros> <mult_count> <threads>

n — matrix dimension (n × n)

pct_zeros — percentage of zeros in the generated matrix (0–100)

mult_count — how many times to repeat the multiplication (useful to amortize measurement noise)

threads — number of OpenMP threads

Example:

./main 1000 90 10 8

This runs with a 1000×1000 matrix with 90% zeros, repeats multiplication 5 times, using 8 threads.

How it works 

The program first generates a random sparse matrix with a specified percentage of zeros.

It builds a CSR representation (arrays v, col_index, row_index).

It computes matrix–vector product using two methods:

Dense representation (standard 2D layout)

CSR (sparse) representation

Each method runs in parallel using OpenMP #pragma omp parallel for.

The program prints elapsed times for building CSR, dense multiplication, and CSR multiplication and verifies that both results match (within 1e-6).

The matrix generator uses random placement of non-zero elements. For very large pct_zeros close to 0 or 100, generation may become slower.
