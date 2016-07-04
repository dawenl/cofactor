'''

Co-factorize the user click matrix and item co-occurrence matrix

CREATED: 2016-03-22 13:21:35 by Dawen Liang <dliang@ee.columbia.edu>

'''

import os
import sys
import time

import numpy as np
from numpy import linalg as LA

from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin

import rec_eval


class CoFacto(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=100, max_iter=10, batch_size=1000,
                 init_std=0.01, dtype='float32', n_jobs=8, random_state=None,
                 save_params=False, save_dir='.', early_stopping=False,
                 verbose=False, **kwargs):
        '''
        CoFacto

        Parameters
        ---------
        n_components : int
            Number of latent factors
        max_iter : int
            Maximal number of iterations to perform
        batch_size: int
            Batch size to perform parallel update
        init_std: float
            The latent factors will be initialized as Normal(0, init_std**2)
        dtype: str or type
            Data-type for the parameters, default 'float32' (np.float32)
        n_jobs: int
            Number of parallel jobs to update latent factors
        random_state : int or RandomState
            Pseudo random number generator used for sampling
        save_params: bool
            Whether to save parameters after each iteration
        save_dir: str
            The directory to save the parameters
        early_stopping: bool
            Whether to early stop the training by monitoring performance on
            validation set
        verbose : bool
            Whether to show progress during model fitting
        **kwargs: dict
            Model hyperparameters
        '''
        self.n_components = n_components
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.init_std = init_std
        self.dtype = dtype
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.save_params = save_params
        self.save_dir = save_dir
        self.early_stopping = early_stopping
        self.verbose = verbose

        if type(self.random_state) is int:
            np.random.seed(self.random_state)
        elif self.random_state is not None:
            np.random.setstate(self.random_state)

        self._parse_kwargs(**kwargs)

    def _parse_kwargs(self, **kwargs):
        ''' Model hyperparameters

        Parameters
        ---------
        lambda_theta, lambda_beta, lambda_gamma: float
            Regularization parameter for user (lambda_theta), item factors (
            lambda_beta), and context factors (lambda_gamma).
        c0, c1: float
            Confidence for 0 and 1 in Hu et al., c0 must be less than c1
        '''
        self.lam_theta = float(kwargs.get('lambda_theta', 1e-5))
        self.lam_beta = float(kwargs.get('lambda_beta', 1e-5))
        self.lam_gamma = float(kwargs.get('lambda_gamma', 1e+0))
        self.c0 = float(kwargs.get('c0', 0.01))
        self.c1 = float(kwargs.get('c1', 1.0))
        assert self.c0 < self.c1, "c0 must be smaller than c1"

    def _init_params(self, n_users, n_items):
        ''' Initialize all the latent factors and biases '''
        self.theta = self.init_std * \
            np.random.randn(n_users, self.n_components).astype(self.dtype)
        self.beta = self.init_std * \
            np.random.randn(n_items, self.n_components).astype(self.dtype)
        self.gamma = self.init_std * \
            np.random.randn(n_items, self.n_components).astype(self.dtype)
        # bias for beta and gamma
        self.bias_b = np.zeros(n_items, dtype=self.dtype)
        self.bias_g = np.zeros(n_items, dtype=self.dtype)
        # global bias
        self.alpha = 0.0

    def fit(self, X, M, F=None, vad_data=None, **kwargs):
        '''Fit the model to the data in X.

        Parameters
        ----------
        X : scipy.sparse.csr_matrix, shape (n_users, n_items)
            Training click matrix.

        M : scipy.sparse.csr_matrix, shape (n_items, n_items)
            Training co-occurrence matrix.

        F : scipy.sparse.csr_matrix, shape (n_items, n_items)
            The weight for the co-occurrence matrix. If not provided,
            weight by default is 1.

        vad_data: scipy.sparse.csr_matrix, shape (n_users, n_items)
            Validation click data. 

        **kwargs: dict
            Additional keywords to evaluation function call on validation data

        Returns
        -------
        self: object
            Returns the instance itself.
        '''
        n_users, n_items = X.shape
        assert M.shape == (n_items, n_items)

        self._init_params(n_users, n_items)
        self._update(X, M, F, vad_data, **kwargs)
        return self

    def transform(self, X):
        pass

    def _update(self, X, M, F, vad_data, **kwargs):
        '''Model training and evaluation on validation set'''
        XT = X.T.tocsr()  # pre-compute this
        self.vad_ndcg = -np.inf
        for i in xrange(self.max_iter):
            if self.verbose:
                print('ITERATION #%d' % i)
            self._update_factors(X, XT, M, F)
            self._update_biases(M, F)
            if vad_data is not None:
                vad_ndcg = self._validate(X, vad_data, **kwargs)
                if self.early_stopping and self.vad_ndcg > vad_ndcg:
                    break  # we will not save the parameter for this iteration
                self.vad_ndcg = vad_ndcg
            if self.save_params:
                self._save_params(i)
        pass

    def _update_factors(self, X, XT, M, F):
        if self.verbose:
            start_t = _writeline_and_time('\tUpdating user factors...')
        self.theta = update_theta(self.beta, X, self.c0,
                                  self.c1, self.lam_theta,
                                  self.n_jobs,
                                  batch_size=self.batch_size)
        if self.verbose:
            print('\r\tUpdating user factors: time=%.2f'
                  % (time.time() - start_t))
            start_t = _writeline_and_time('\tUpdating item factors...')
        self.beta = update_beta(self.theta, self.gamma,
                                self.bias_b, self.bias_g, self.alpha,
                                XT, M, F, self.c0, self.c1, self.lam_beta,
                                self.n_jobs,
                                batch_size=self.batch_size)
        if self.verbose:
            print('\r\tUpdating item factors: time=%.2f'
                  % (time.time() - start_t))
            start_t = _writeline_and_time('\tUpdating context factors...')
        # here it really should be M^T and F^T, but both are symmetric
        self.gamma = update_gamma(self.beta, self.bias_b, self.bias_g,
                                  self.alpha, M, F, self.lam_gamma,
                                  self.n_jobs,
                                  batch_size=self.batch_size)
        if self.verbose:
            print('\r\tUpdating context factors: time=%.2f'
                  % (time.time() - start_t))
        pass

    def _update_biases(self, M, F):
        if self.verbose:
            start_t = _writeline_and_time('\tUpdating bias terms...')
        self.bias_b = update_bias(self.beta, self.gamma,
                                  self.bias_g, self.alpha, M, F,
                                  self.n_jobs, batch_size=self.batch_size)
        # here it really should be M^T and F^T, but both are symmetric
        self.bias_g = update_bias(self.gamma, self.beta,
                                  self.bias_b, self.alpha, M, F,
                                  self.n_jobs, batch_size=self.batch_size)
        self.alpha = update_alpha(self.beta, self.gamma,
                                  self.bias_b, self.bias_g, M, F,
                                  self.n_jobs, batch_size=self.batch_size)
        if self.verbose:
            print('\r\tUpdating bias terms: time=%.2f'
                  % (time.time() - start_t))
        pass

    def _validate(self, X, vad_data, **kwargs):
        vad_ndcg = rec_eval.normalized_dcg_at_k(X, vad_data,
                                                self.theta,
                                                self.beta,
                                                **kwargs)
        if self.verbose:
            print('\tValidation NDCG@k: %.5f' % vad_ndcg)
        return vad_ndcg

    def _save_params(self, iter):
        '''Save the parameters'''
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        filename = 'CoFacto_K%d_iter%d.npz' % (self.n_components, iter)
        np.savez(os.path.join(self.save_dir, filename), U=self.theta,
                 V=self.beta)


# Utility functions #
def _writeline_and_time(s):
    sys.stdout.write(s)
    sys.stdout.flush()
    return time.time()


def get_row(Y, i):
    '''Given a scipy.sparse.csr_matrix Y, get the values and indices of the
    non-zero values in i_th row'''
    lo, hi = Y.indptr[i], Y.indptr[i + 1]
    return Y.data[lo:hi], Y.indices[lo:hi]


def update_theta(beta, X, c0, c1, lam_theta, n_jobs, batch_size=1000):
    '''Update user latent factors'''
    m, n = X.shape  # m: number of users, n: number of items
    f = beta.shape[1]  # f: number of factors

    BTB = c0 * np.dot(beta.T, beta)  # precompute this
    BTBpR = BTB + lam_theta * np.eye(f, dtype=beta.dtype)

    start_idx = range(0, m, batch_size)
    end_idx = start_idx[1:] + [m]
    res = Parallel(n_jobs=n_jobs)(
        delayed(_solve_weighted_factor)(
            lo, hi, beta, X, BTBpR, c0, c1, f, lam_theta)
        for lo, hi in zip(start_idx, end_idx))
    theta = np.vstack(res)
    return theta


def _solve_weighted_factor(lo, hi, beta, X, BTBpR, c0, c1, f, lam_theta):
    theta_batch = np.empty((hi - lo, f), dtype=beta.dtype)
    for ib, u in enumerate(xrange(lo, hi)):
        x_u, idx_u = get_row(X, u)
        B_u = beta[idx_u]
        a = x_u.dot(c1 * B_u)
        B = BTBpR + B_u.T.dot((c1 - c0) * B_u)
        theta_batch[ib] = LA.solve(B, a)
    return theta_batch


def update_beta(theta, gamma, bias_b, bias_g, alpha, XT, M, F, c0, c1,
                lam_beta, n_jobs, batch_size=1000):
    '''Update item latent factors/embeddings'''
    n, m = XT.shape  # m: number of users, n: number of items
    f = theta.shape[1]
    assert theta.shape[0] == m
    assert gamma.shape == (n, f)

    TTT = c0 * np.dot(theta.T, theta)  # precompute this
    TTTpR = TTT + lam_beta * np.eye(f, dtype=theta.dtype)

    start_idx = range(0, n, batch_size)
    end_idx = start_idx[1:] + [n]
    res = Parallel(n_jobs=n_jobs)(
        delayed(_solve_weighted_cofactor)(
            lo, hi, theta, gamma, bias_b, bias_g, alpha, XT, M, F, TTTpR, c0,
            c1, f, lam_beta)
        for lo, hi in zip(start_idx, end_idx))
    beta = np.vstack(res)
    return beta


def _solve_weighted_cofactor(lo, hi, theta, gamma, bias_b, bias_g, alpha, XT,
                             M, F, TTTpR, c0, c1, f, lam_beta):
    beta_batch = np.empty((hi - lo, f), dtype=theta.dtype)
    for ib, i in enumerate(xrange(lo, hi)):
        x_i, idx_x_i = get_row(XT, i)
        T_i = theta[idx_x_i]

        m_i, idx_m_i = get_row(M, i)
        G_i = gamma[idx_m_i]

        rsd = m_i - bias_b[i] - bias_g[idx_m_i] - alpha

        if F is not None:
            f_i, _ = get_row(F, i)
            GTG = G_i.T.dot(G_i * f_i[:, np.newaxis])
            rsd *= f_i
        else:
            GTG = G_i.T.dot(G_i)

        B = TTTpR + T_i.T.dot((c1 - c0) * T_i) + GTG
        a = x_i.dot(c1 * T_i) + np.dot(rsd, G_i)
        beta_batch[ib] = LA.solve(B, a)
    return beta_batch


def update_gamma(beta, bias_b, bias_g, alpha, MT, FT, lam_gamma,
                 n_jobs, batch_size=1000):
    '''Update context latent factors'''
    n, f = beta.shape  # n: number of items, f: number of factors

    start_idx = range(0, n, batch_size)
    end_idx = start_idx[1:] + [n]
    res = Parallel(n_jobs=n_jobs)(
        delayed(_solve_factor)(
            lo, hi, beta, bias_b, bias_g, alpha, MT, FT, f, lam_gamma)
        for lo, hi in zip(start_idx, end_idx))
    gamma = np.vstack(res)
    return gamma


def _solve_factor(lo, hi, beta, bias_b, bias_g, alpha, MT, FT, f, lam_gamma,
                  BTBpR=None):
    gamma_batch = np.empty((hi - lo, f), dtype=beta.dtype)
    for ib, j in enumerate(xrange(lo, hi)):
        m_j, idx_j = get_row(MT, j)
        rsd = m_j - bias_b[idx_j] - bias_g[j] - alpha
        B_j = beta[idx_j]
        if FT is not None:
            f_j, _ = get_row(FT, j)
            BTB = B_j.T.dot(B_j * f_j[:, np.newaxis])
            rsd *= f_j
        else:
            BTB = B_j.T.dot(B_j)

        B = BTB + lam_gamma * np.eye(f, dtype=beta.dtype)
        a = np.dot(rsd, B_j)
        gamma_batch[ib] = LA.solve(B, a)
    return gamma_batch


def update_bias(beta, gamma, bias_g, alpha, M, F, n_jobs, batch_size=1000):
    ''' Update the per-item (or context) bias term.
    '''
    n = beta.shape[0]

    start_idx = range(0, n, batch_size)
    end_idx = start_idx[1:] + [n]

    res = Parallel(n_jobs=n_jobs)(
        delayed(_solve_bias)(lo, hi, beta, gamma, bias_g, alpha, M, F)
        for lo, hi in zip(start_idx, end_idx))
    bias_b = np.hstack(res)
    return bias_b


def _solve_bias(lo, hi, beta, gamma, bias_g, alpha, M, F):
    bias_b_batch = np.empty(hi - lo, dtype=beta.dtype)
    for ib, i in enumerate(xrange(lo, hi)):
        m_i, idx_i = get_row(M, i)
        m_i_hat = gamma[idx_i].dot(beta[i]) + bias_g[idx_i] + alpha
        rsd = m_i - m_i_hat

        if F is not None:
            f_i, _ = get_row(F, i)
            rsd *= f_i

        if rsd.size > 0:
            bias_b_batch[ib] = rsd.mean()
        else:
            bias_b_batch[ib] = 0.
    return bias_b_batch


def update_alpha(beta, gamma, bias_b, bias_g, M, F, n_jobs, batch_size=1000):
    ''' Update the global bias term
    '''
    n = beta.shape[0]
    assert beta.shape == gamma.shape
    assert bias_b.shape == bias_g.shape

    start_idx = range(0, n, batch_size)
    end_idx = start_idx[1:] + [n]

    res = Parallel(n_jobs=n_jobs)(
        delayed(_solve_alpha)(lo, hi, beta, gamma, bias_b, bias_g, M, F)
        for lo, hi in zip(start_idx, end_idx))

    return np.sum(res) / M.data.size


def _solve_alpha(lo, hi, beta, gamma, bias_b, bias_g, M, F):
    res = 0.
    for ib, i in enumerate(xrange(lo, hi)):
        m_i, idx_i = get_row(M, i)
        m_i_hat = gamma[idx_i].dot(beta[i]) + bias_b[i] + bias_g[idx_i]
        rsd = m_i - m_i_hat

        if F is not None:
            f_i, _ = get_row(F, i)
            rsd *= f_i
        res += rsd.sum()
    return res
