import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np

from joblib import Parallel, delayed


def get_row(S, i):
    lo, hi = S.indptr[i], S.indptr[i + 1]
    return S.data[lo:hi], S.indices[lo:hi]


def solve_sequential(As, Bs):
    X_stack = np.empty_like(As, dtype=As.dtype)

    for k in xrange(As.shape[0]):
        X_stack[k] = np.linalg.solve(Bs[k], As[k])

    return X_stack


def solve_batch(b, S, Y, X_reg, YTYpR, batch_size, m, f, dtype):
    lo = b * batch_size
    hi = min((b + 1) * batch_size, m)
    current_batch_size = hi - lo

    A_stack = np.empty((current_batch_size, f), dtype=dtype)
    B_stack = np.empty((current_batch_size, f, f), dtype=dtype)

    for ib, k in enumerate(xrange(lo, hi)):
        s_u, i_u = get_row(S, k)

        Y_u = Y[i_u]  # exploit sparsity
        A = (s_u + 1).dot(Y_u)

        if X_reg is not None:
            A += X_reg[k]

        YTSY = np.dot(Y_u.T, (Y_u * s_u[:, None]))
        B = YTSY + YTYpR

        A_stack[ib] = A
        B_stack[ib] = B

    X_stack = solve_sequential(A_stack, B_stack)
    return X_stack


def recompute_factors_batched(Y, S, lambda_reg, X=None,
                              dtype='float32', batch_size=2000, n_jobs=20):
    m = S.shape[0]  # m = number of users
    f = Y.shape[1]  # f = number of factors

    YTY = np.dot(Y.T, Y)  # precompute this
    YTYpR = YTY + lambda_reg * np.eye(f)
    if X is not None:
        X_reg = lambda_reg * X
    else:
        X_reg = None
    X_new = np.zeros((m, f), dtype=dtype)

    num_batches = int(np.ceil(m / float(batch_size)))

    res = Parallel(n_jobs=n_jobs)(delayed(solve_batch)(b, S, Y, X_reg, YTYpR,
                                                       batch_size, m, f, dtype)
                                  for b in xrange(num_batches))
    X_new = np.concatenate(res, axis=0)

    return X_new
