import sys
import time

import numpy as np
from scipy import weave

import batched_inv_joblib
import rec_eval


def linear_surplus_confidence_matrix(B, alpha):
    # To construct the surplus confidence matrix, we need to operate only on
    # the nonzero elements.
    # This is not possible: S = alpha * B
    S = B.copy()
    S.data = alpha * S.data
    return S


def log_surplus_confidence_matrix(B, alpha, epsilon):
    # To construct the surplus confidence matrix, we need to operate only on
    # the nonzero elements.
    # This is not possible: S = alpha * np.log(1 + B / epsilon)
    S = B.copy()
    S.data = alpha * np.log(1 + S.data / epsilon)
    return S


def factorize(S, num_factors, X=None, vad_data=None, num_iters=10, init_std=0.01,
              lambda_U_reg=1e-2, lambda_V_reg=100, lambda_W_reg=1e-2,
              dtype='float32', random_state=None, verbose=False,
              recompute_factors=batched_inv_joblib.recompute_factors_batched,
              fixed_item_embeddings=False,
              V=None,
              *args, **kwargs):

    num_users, num_items = S.shape
    if X is not None:
        assert X.shape == (num_items, num_factors)

    if verbose:
        print "Precompute S^T (if necessary)"
        start_time = time.time()

    ST = S.T.tocsr()

    if verbose:
        print "  took %.3f seconds" % (time.time() - start_time)
        start_time = time.time()

    if type(random_state) is int:
        np.random.seed(random_state)
    elif random_state is not None:
        np.random.setstate(random_state)

    U = None
    if not fixed_item_embeddings and not V:
        V = np.random.randn(num_items, num_factors).astype(dtype) * init_std

    old_ndcg = -np.inf
    for i in xrange(num_iters):
        if verbose:
            print("Iteration %d:" % i)
            start_t = _write_and_time('\tUpdating user factors...')
        U = recompute_factors(V, S, lambda_U_reg, dtype=dtype, *args, **kwargs)

        if verbose:
            print('\r\tUpdating user factors: time=%.2f'
                  % (time.time() - start_t))
            if not fixed_item_embeddings:
                start_t = _write_and_time('\tUpdating item factors...')
        if not fixed_item_embeddings:
            V = recompute_factors(U, ST, lambda_V_reg, X=X, dtype=dtype,
                                  *args, **kwargs)
        if verbose and not fixed_item_embeddings:
            print('\r\tUpdating item factors: time=%.2f'
                  % (time.time() - start_t))
        if vad_data is not None and not fixed_item_embeddings:
            vad_ndcg = rec_eval.normalized_dcg_at_k(S, vad_data, U, V,
                                                    k=100,
                                                    batch_users=5000)
            if verbose:
                print("\tValidation NDCG@k: %.5f" % vad_ndcg)
                sys.stdout.flush()
            if old_ndcg > vad_ndcg:
                break  # we will not save the parameter for this iteration
            old_ndcg = vad_ndcg

    return U, V, old_ndcg


def _pred_loglikeli(U, V, dtype, X_new=None, rows_new=None, cols_new=None):
    X_pred = _inner(U, V, rows_new, cols_new, dtype)
    pred_ll = np.mean((X_new.data - X_pred)**2)
    return pred_ll


def _write_and_time(s):
    sys.stdout.write(s)
    sys.stdout.flush()
    return time.time()


def _inner(U, V, rows, cols, dtype):
    n_ratings = rows.size
    n_components = U.shape[1]
    assert V.shape[1] == n_components
    data = np.empty(n_ratings, dtype=dtype)
    code = r"""
    for (int i = 0; i < n_ratings; i++) {
       data[i] = 0.0;
       for (int j = 0; j < n_components; j++) {
           data[i] += U[rows[i] * n_components + j] * V[cols[i] * n_components + j];
       }
    }
    """
    weave.inline(code, ['data', 'U', 'V', 'rows', 'cols', 'n_ratings',
                        'n_components'])
    return data
