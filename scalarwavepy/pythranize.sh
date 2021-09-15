#!/bin/sh

pythran -DUSE_XSIMD -fopenmp -march=native -o pythran_lax.so pythran_lax.py

# def solve(u, nx, nt, c, num_threads=8):
#     i=0
#     j=0
#     for j in range(0, nt):
#         #pragma omp parallel num_threads(num_threads) firstprivate(nx,j) private(i) shared(u,c) default(none)
#         #pragma omp for
#         for i in range(1, nx):
#             u[i, j+1] = u[i,j] - (c/2.)*(u[i+1,j] - u[i-1,j]) + ((c*c)/2) * (u[i+1,j] - 2. * u[i,j] + u[i-1,j] )
#     return u
