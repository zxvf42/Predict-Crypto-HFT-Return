import numpy as np
import numba as nb

# Util funcs
jit = nb.njit(error_model="numpy", cache=True)
pjit = nb.njit(error_model="numpy", parallel=True, cache=True)

@jit
def log(v):
    return np.log(v)

@jit
def log1p(v):
    return np.log1p(v)

@jit
def max(v):
    return np.max(v)


@jit
def min(v):
    return np.min(v)


@jit
def median(v):
    return np.median(v)


@jit
def sign(v):
    return np.sign(v)


@jit
def mean(v):
    if len(v) == 0:
        return 0
    return np.mean(v)

@jit
def sqrt(v):
    return np.sqrt(v)

@jit
def std(v):
    return np.std(v)


@jit
def var(v):
    return np.var(v)


@jit
def sum(v):
    return np.sum(v)


@jit
def cumsum(v):
    return np.cumsum(v)


@jit
def abs(v):
    return np.abs(v)


@jit
def ptp(v):
    return np.ptp(v)


@jit
def diff(v, offset=1):
    dtype = v.dtype
    if len(v) > offset:
        return v[offset:] - v[:-offset]
    else:
        return np.zeros(1, dtype=dtype)