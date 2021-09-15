#pythran export solve(float64 [:,:], int64, int64, float64)
#pythran export solve(float64 [:,:], int64, int64, float64, int64)


import numpy as __pythran_import_numpy
def solve(u, nx, nt, c, num_threads=8):
    for j_ in builtins.range(0, nt):
        for i_ in builtins.range(1, nx):
            u[(i_, (j_ + 1))] = ((u[(i_, j_)] - ((c / 2.0) * (u[((i_ + 1), j_)] - u[((i_ - 1), j_)]))) + ((__pythran_import_numpy.square(c) / 2) * ((u[((i_ + 1), j_)] - (2.0 * u[(i_, j_)])) + u[((i_ - 1), j_)])))
    return u
