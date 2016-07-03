import bottleneck as bn
import numpy as np

from scipy import sparse


"""
All the data should be in the shape of (n_users, n_items)
All the latent factors should in the shape of (n_users/n_items, n_components)

1. train_data refers to the data that was used to train the model
2. heldout_data refers to the data that was used for evaluation (could be test
set or validation set)
3. vad_data refers to the data that should be excluded as validation set, which
should only be used when calculating test scores

"""


def prec_at_k(train_data, heldout_data, U, V, batch_users=5000, k=20,
              mu=None, vad_data=None, agg=np.nanmean):
    n_users = train_data.shape[0]
    res = list()
    for user_idx in user_idx_generator(n_users, batch_users):
        res.append(precision_at_k_batch(train_data, heldout_data,
                                        U, V.T, user_idx, k=k,
                                        mu=mu, vad_data=vad_data))
    mn_prec = np.hstack(res)
    if callable(agg):
        return agg(mn_prec)
    return mn_prec


def recall_at_k(train_data, heldout_data, U, V, batch_users=5000, k=20,
                mu=None, vad_data=None, agg=np.nanmean):
    n_users = train_data.shape[0]
    res = list()
    for user_idx in user_idx_generator(n_users, batch_users):
        res.append(recall_at_k_batch(train_data, heldout_data,
                                     U, V.T, user_idx, k=k,
                                     mu=mu, vad_data=vad_data))
    mn_recall = np.hstack(res)
    if callable(agg):
        return agg(mn_recall)
    return mn_recall


def ric_rank_at_k(train_data, heldout_data, U, V, batch_users=5000, k=5,
                  mu=None, vad_data=None):
    n_users = train_data.shape[0]
    res = list()
    for user_idx in user_idx_generator(n_users, batch_users):
        res.append(mean_rrank_at_k_batch(train_data, heldout_data,
                                         U, V.T, user_idx, k=k,
                                         mu=mu, vad_data=vad_data))
    mrrank = np.hstack(res)
    return mrrank[mrrank > 0].mean()


def mean_perc_rank(train_data, heldout_data, U, V, batch_users=5000,
                   mu=None, vad_data=None):
    n_users = train_data.shape[0]
    mpr = 0
    for user_idx in user_idx_generator(n_users, batch_users):
        mpr += mean_perc_rank_batch(train_data, heldout_data, U, V.T, user_idx,
                                    mu=mu, vad_data=vad_data)
    mpr /= heldout_data.sum()
    return mpr


def normalized_dcg(train_data, heldout_data, U, V, batch_users=5000,
                   mu=None, vad_data=None, agg=np.nanmean):
    n_users = train_data.shape[0]
    res = list()
    for user_idx in user_idx_generator(n_users, batch_users):
        res.append(NDCG_binary_batch(train_data, heldout_data, U, V.T,
                                     user_idx, mu=mu, vad_data=vad_data))
    ndcg = np.hstack(res)
    if callable(agg):
        return agg(ndcg)
    return ndcg


def normalized_dcg_at_k(train_data, heldout_data, U, V, batch_users=5000,
                        k=100, mu=None, vad_data=None, agg=np.nanmean):

    n_users = train_data.shape[0]
    res = list()
    for user_idx in user_idx_generator(n_users, batch_users):
        res.append(NDCG_binary_at_k_batch(train_data, heldout_data, U, V.T,
                                          user_idx, k=k, mu=mu,
                                          vad_data=vad_data))
    ndcg = np.hstack(res)
    if callable(agg):
        return agg(ndcg)
    return ndcg


def map_at_k(train_data, heldout_data, U, V, batch_users=5000, k=100, mu=None,
             vad_data=None, agg=np.nanmean):

    n_users = train_data.shape[0]
    res = list()
    for user_idx in user_idx_generator(n_users, batch_users):
        res.append(MAP_at_k_batch(train_data, heldout_data, U, V.T, user_idx,
                                  k=k, mu=mu, vad_data=vad_data))
    map = np.hstack(res)
    if callable(agg):
        return agg(map)
    return map


# helper functions #

def user_idx_generator(n_users, batch_users):
    ''' helper function to generate the user index to loop through the dataset
    '''
    for start in xrange(0, n_users, batch_users):
        end = min(n_users, start + batch_users)
        yield slice(start, end)


def _make_prediction(train_data, Et, Eb, user_idx, batch_users, mu=None,
                     vad_data=None):
    n_songs = train_data.shape[1]
    # exclude examples from training and validation (if any)
    item_idx = np.zeros((batch_users, n_songs), dtype=bool)
    item_idx[train_data[user_idx].nonzero()] = True
    if vad_data is not None:
        item_idx[vad_data[user_idx].nonzero()] = True
    X_pred = Et[user_idx].dot(Eb)
    if mu is not None:
        if isinstance(mu, np.ndarray):
            assert mu.size == n_songs  # mu_i
            X_pred *= mu
        elif isinstance(mu, dict):  # func(mu_ui)
            params, func = mu['params'], mu['func']
            args = [params[0][user_idx], params[1]]
            if len(params) > 2:  # for bias term in document or length-scale
                args += [params[2][user_idx]]
            if not callable(func):
                raise TypeError("expecting a callable function")
            X_pred *= func(*args)
        else:
            raise ValueError("unsupported mu type")
    X_pred[item_idx] = -np.inf
    return X_pred


def precision_at_k_batch(train_data, heldout_data, Et, Eb, user_idx,
                         k=20, normalize=True, mu=None, vad_data=None):
    batch_users = user_idx.stop - user_idx.start

    X_pred = _make_prediction(train_data, Et, Eb, user_idx,
                              batch_users, mu=mu, vad_data=vad_data)
    idx = bn.argpartsort(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_data[user_idx] > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)

    if normalize:
        precision = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    else:
        precision = tmp / k
    return precision


def recall_at_k_batch(train_data, heldout_data, Et, Eb, user_idx,
                      k=20, normalize=True, mu=None, vad_data=None):
    batch_users = user_idx.stop - user_idx.start

    X_pred = _make_prediction(train_data, Et, Eb, user_idx,
                              batch_users, mu=mu, vad_data=vad_data)
    idx = bn.argpartsort(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_data[user_idx] > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall


def mean_rrank_at_k_batch(train_data, heldout_data, Et, Eb,
                          user_idx, k=5, mu=None, vad_data=None):
    '''
    mean reciprocal rank@k: For each user, make predictions and rank for
    all the items. Then calculate the mean reciprocal rank for the top K that
    are in the held-out set.
    '''
    batch_users = user_idx.stop - user_idx.start

    X_pred = _make_prediction(train_data, Et, Eb, user_idx,
                              batch_users, mu=mu, vad_data=vad_data)
    all_rrank = 1. / (np.argsort(np.argsort(-X_pred, axis=1), axis=1) + 1)
    X_true_binary = (heldout_data[user_idx] > 0).toarray()

    heldout_rrank = X_true_binary * all_rrank
    top_k = bn.partsort(-heldout_rrank, k, axis=1)
    return -top_k[:, :k].mean(axis=1)


def NDCG_binary_batch(train_data, heldout_data, Et, Eb, user_idx,
                      mu=None, vad_data=None):
    '''
    normalized discounted cumulative gain for binary relevance
    '''
    batch_users = user_idx.stop - user_idx.start
    n_items = train_data.shape[1]

    X_pred = _make_prediction(train_data, Et, Eb, user_idx,
                              batch_users, mu=mu, vad_data=vad_data)
    all_rank = np.argsort(np.argsort(-X_pred, axis=1), axis=1)
    # build the discount template
    tp = 1. / np.log2(np.arange(2, n_items + 2))
    all_disc = tp[all_rank]

    X_true_binary = (heldout_data[user_idx] > 0).tocoo()
    disc = sparse.csr_matrix((all_disc[X_true_binary.row, X_true_binary.col],
                              (X_true_binary.row, X_true_binary.col)),
                             shape=all_disc.shape)
    DCG = np.array(disc.sum(axis=1)).ravel()
    IDCG = np.array([tp[:n].sum()
                     for n in heldout_data[user_idx].getnnz(axis=1)])
    return DCG / IDCG


def NDCG_binary_at_k_batch(train_data, heldout_data, Et, Eb, user_idx,
                           mu=None, k=100, vad_data=None):
    '''
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = user_idx.stop - user_idx.start

    X_pred = _make_prediction(train_data, Et, Eb, user_idx,
                              batch_users, mu=mu, vad_data=vad_data)
    idx_topk_part = bn.argpartsort(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    heldout_batch = heldout_data[user_idx]
    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].toarray() * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)])
    return DCG / IDCG


def MAP_at_k_batch(train_data, heldout_data, Et, Eb, user_idx, mu=None, k=100,
                   vad_data=None):
    '''
    mean average precision@k
    '''
    batch_users = user_idx.stop - user_idx.start

    X_pred = _make_prediction(train_data, Et, Eb, user_idx, batch_users, mu=mu,
                              vad_data=vad_data)
    idx_topk_part = bn.argpartsort(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]

    aps = np.zeros(batch_users)
    for i, idx in enumerate(xrange(user_idx.start, user_idx.stop)):
        actual = heldout_data[idx].nonzero()[1]
        if len(actual) > 0:
            predicted = idx_topk[i]
            aps[i] = apk(actual, predicted, k=k)
        else:
            aps[i] = np.nan
    return aps


def mean_perc_rank_batch(train_data, heldout_data, Et, Eb, user_idx,
                         mu=None, vad_data=None):
    '''
    mean percentile rank for a batch of users
    MPR of the full set is the sum of batch MPR's divided by the sum of all the
    feedbacks. (Eq. 8 in Hu et al.)
    This metric not necessarily constrains the data to be binary
    '''
    batch_users = user_idx.stop - user_idx.start

    X_pred = _make_prediction(train_data, Et, Eb, user_idx, batch_users,
                              mu=mu, vad_data=vad_data)
    all_perc = np.argsort(np.argsort(-X_pred, axis=1), axis=1) / \
        np.isfinite(X_pred).sum(axis=1, keepdims=True).astype(np.float32)
    perc_batch = (all_perc[heldout_data[user_idx].nonzero()] *
                  heldout_data[user_idx].data).sum()
    return perc_batch


## steal from https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
def apk(actual, predicted, k=100):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual: #and p not in predicted[:i]: # not necessary for us since we will not make duplicated recs
            num_hits += 1.0
            score += num_hits / (i+1.0)

    # we handle this part before making the function call
    #if not actual:
    #    return np.nan

    return score / min(len(actual), k)
