"""*****************************************************************************************
MIT License

Copyright (c) 2019 Alaa Maalouf, Ibrahim Jubran, Dan Feldman

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*****************************************************************************************"""


####################################### NOTES ############################################
# - Please cite our paper when using the code: 
#             "Fast and Accurate Least-Mean-Squares Solvers" (NIPS19' - Oral presentation) 
#                          Alaa Maalouf and Ibrahim Jubran and Dan Feldman
#
# - Faster algorithm for large values of the dimension d will be published soon.
##########################################################################################


import numpy as np
from sklearn import linear_model
from sklearn.model_selection import KFold
import time
import math

def Caratheodory(P, u, dtype='float64'):
    """
    Implementation of the Caratheodory Theorem(1907)
    input: a numpy array P containing n rows (points), each of size d, and a positive vector of weights u (that sums to 1)
    output:a new vector of weights new_u that satisfies :
                1. new_u is positive and sums to 1
                2. new_u has at most d+1 non zero entries
                3. the weighted sum of P and u (input) is the same as the weighted sum of P and new_u (output)
    computation time: O(n^2d^2)
    """
    while 1:
        n = np.count_nonzero(u)
        d = P.shape[1]
        u_non_zero = np.nonzero(u)

        if n <= d + 1:
            return u

        A = P[u_non_zero]
        reduced_vec = np.outer(A[0], np.ones(A.shape[0]-1, dtype = dtype))
        A = A[1:].T - reduced_vec

        _, _, V = np.linalg.svd(A, full_matrices=True)
        v=V[-1]
        v = np.insert(v, [0],   -1 * np.sum(v))

        idx_good_alpha = np.nonzero(v > 0)
        alpha = np.min(u[u_non_zero][idx_good_alpha]/v[idx_good_alpha])

        w = np.zeros(u.shape[0] , dtype = dtype)
        tmp_w = u[u_non_zero] - alpha * v
        tmp_w[np.argmin(tmp_w)] = 0.0
        w[u_non_zero] = tmp_w
        w[u_non_zero][np.argmin(w[u_non_zero] )] = 0
        u = w

def Fast_Caratheodory(P,u,coreset_size, dtype = 'float64'):
    """
    Our fast and accurate implementation of Caratheodory's Theorem
    Input: a numpy array P containing n rows (points), each of size d, and a positive vector of weights u (if u does not
    sum to 1, we first normalize u by its sum, then multiply u back by its original sum before returning it)
    Output: a new vector of positive weights new_u that satisfies :
                 1. new_u has at most d+1 non zero entries
                 2. the weighted sum of P and u (input) is the same as the weighted sum of P and new_u (output)
    Computation time: O(nd+logn*d^4)
    """
    d = P.shape[1]
    n = P.shape[0]
    m = 2*d + 2
    if n <= d + 1:
        return u.reshape(-1)

    u_sum = np.sum(u)
    u = u/u_sum
    chunk_size = math.ceil(n/m)
    current_m = math.ceil(n/chunk_size)

    add_z = chunk_size - int (n%chunk_size)
    u = u.reshape(-1,1)
    if add_z != chunk_size:
        zeros = np.zeros((add_z, P.shape[1]), dtype = dtype)
        P = np.concatenate((P, zeros))
        zeros = np.zeros((add_z, u.shape[1]), dtype = dtype)
        u = np.concatenate((u, zeros))
    
    idxarray = np.array(range(P.shape[0]) )
    
    p_groups = P.reshape(current_m, chunk_size, P.shape[1])
    u_groups = u.reshape(current_m, chunk_size)
    idx_group = idxarray.reshape(current_m, chunk_size)
    u_nonzero = np.count_nonzero(u)

    if not coreset_size:
        coreset_size = d+1
    while u_nonzero > coreset_size:

        groups_means = np.einsum('ijk,ij->ik',p_groups, u_groups)
        group_weigts = np.ones(groups_means.shape[0], dtype = dtype)*1/current_m

        Cara_u_idx = Caratheodory(groups_means , group_weigts,dtype = dtype )

        IDX = np.nonzero(Cara_u_idx)

        new_P = p_groups[IDX].reshape(-1,d)

        subset_u = (current_m * u_groups[IDX] * Cara_u_idx[IDX][:, np.newaxis]).reshape(-1, 1)
        new_idx_array = idx_group[IDX].reshape(-1,1)
        ##############################################################################3
        u_nonzero = np.count_nonzero(subset_u)
        chunk_size = math.ceil(new_P.shape[0]/ m)
        current_m = math.ceil(new_P.shape[0]/ chunk_size)

        add_z = chunk_size - int(new_P.shape[0] % chunk_size)
        if add_z != chunk_size:
            new_P = np.concatenate((new_P, np.zeros((add_z, new_P.shape[1]), dtype = dtype)))
            subset_u = np.concatenate((subset_u, np.zeros((add_z, subset_u.shape[1]),dtype = dtype)))
            new_idx_array = np.concatenate((new_idx_array, np.zeros((add_z, new_idx_array.shape[1]),dtype = dtype)))
        p_groups = new_P.reshape(current_m, chunk_size, new_P.shape[1])
        u_groups = subset_u.reshape(current_m, chunk_size)
        idx_group = new_idx_array.reshape(current_m , chunk_size)
        ###########################################################

    new_u = np.zeros(n)
    subset_u = subset_u[(new_idx_array < n)]
    new_idx_array = new_idx_array[(new_idx_array < n)].reshape(-1).astype(int)
    new_u[new_idx_array] = subset_u
    return u_sum * new_u


def linregcoreset(P, u, b=None, c_size=None, dtype='float64'):
    """
    This function computes a coreset for linear regression.
    Input: a numpy array P containing n rows (points), each of size d, a positive vector of weights u of size n, a labels
           vector b of size n, coreset size c_size (not required).
    Output: a new numpy array new_P containing the coreset points in its rows and a new vector new_u of positive weights,
            and a new vector of labels new_b for the coreset. The output satisfies for every vector x that:
                 ||sqrt(u.transpose())*(Px-b)||^2 = ||sqrt(new_u.transpose())*(new_Px-new_b)||^2
                 i.e., the output of a call to linearRegression with the original input or with the coreset is the same.
    Computation time: O(nd^2+logn*d^8)
    """
    if b is not None:
        P_tag = np.append(P, b, axis=1)
    else:
        P_tag = P

    n_tag = P_tag.shape[0]; d_tag = P_tag.shape[1]
    P_tag = P_tag.reshape(n_tag, d_tag, 1)
    
    P_tag = np.einsum("ikj,ijk->ijk",P_tag ,P_tag)
    P_tag = P_tag.reshape(n_tag, -1)
    n_tag = P_tag.shape[0]; d_tag = P_tag.shape[1]

    coreset_weigts = Fast_Caratheodory(P_tag.reshape(n_tag,-1), u, c_size,  dtype=dtype)
    new_idx_array = np.nonzero(coreset_weigts)
    coreset_weigts = coreset_weigts[new_idx_array]

    if b is not None:
        return P[new_idx_array], coreset_weigts.reshape(-1), b[new_idx_array]
    else:
        return P[new_idx_array], coreset_weigts.reshape(-1)


def stream_coreset(P, u, b, folds=None, dtype='float64'):
    """
    This function computes a coreset for LMS solvers that use k-fold cross validation. It partitions the data into "folds"
    parts, and computes a coreset for every part using the function linregcoreset.
    Input: a numpy array P containing n rows (points), each of size d, a positive vector of weights u of size n, a labels
           vector b of size n, and the number of folds used in the cross validation.
    Output: a new numpy array new_P containing the coreset points in its rows and a new vector new_u of positive weights,
            and a new vector of labels new_b for the coreset. The output satisfies for every vector x that:
                 ||sqrt(u.transpose())*(Px-b)||^2 = ||sqrt(new_u.transpose())*(new_Px-new_b)||^2
                 i.e., the output of a call to linearRegression with the original input or with the coreset is the same.
    Computation time: O(nd^2+logn*d^8)
    """
    if folds is None:
        return linregcoreset(P, u, b, dtype=dtype)
    m = int(P.shape[0] / folds)

    d = P.shape[1]
    size_of_coreset = ((d+1)*(d+1)+1)

    batches = folds
    cc, uc, bc = linregcoreset(P[0:m], u[0:m], b[0:m], dtype=dtype)

    if cc.shape[0] < size_of_coreset and folds:
            add_z = size_of_coreset - cc.shape[0]
            zeros = np.zeros((add_z, cc.shape[1]), dtype=dtype)
            cc = np.concatenate((cc, zeros))
            zeros = np.zeros((add_z), dtype=dtype)
            uc = np.concatenate((uc, zeros))
            zeros = np.zeros((add_z, bc.shape[1]), dtype=dtype)
            bc = np.concatenate((bc, zeros))

    for batch in range(1, batches):
        coreset, new_u, new_b = linregcoreset(P[batch*m:(batch+1)*m], u[batch*m:(batch+1)*m], b[batch*m:(batch+1)*m], dtype=dtype)

        if coreset.shape[0] < size_of_coreset and folds:
            add_z = size_of_coreset - coreset.shape[0]
            zeros = np.zeros((add_z, coreset.shape[1]), dtype=dtype)
            coreset = np.concatenate((coreset, zeros))
            zeros = np.zeros((add_z),dtype=dtype)
            new_u = np.concatenate((new_u, zeros))
            zeros = np.zeros((add_z, new_b.shape[1]), dtype=dtype)
            new_b = np.concatenate((new_b, zeros))
        bc = np.concatenate((bc, new_b))
        cc = np.concatenate((cc, coreset))
        uc = np.concatenate((uc, new_u))
    return cc, uc, bc

###################################################################################


def test_model(test_data, test_labels, test_weights, clf):
    weighted_test_data = test_data * np.sqrt(test_weights[:, np.newaxis])
    weighted_test_labels = test_labels * np.sqrt(test_weights[:, np.newaxis])
    score = clf.score(weighted_test_data, weighted_test_labels)
    return score


def train_model(data, labels, weights, clf):
    time_start = time.time()
    weighted_data = data * np.sqrt(weights[:, np.newaxis])
    weighted_labels = (labels * np.sqrt(weights[:, np.newaxis])).ravel()
    clf.fit(weighted_data, weighted_labels)
    time_end = time.time()

    return time_end - time_start, clf


def coreset_train_model(data, labels, weights, clf, folds=None, solver='ridge'):
    time_start = time.time()

    coreset, coreset_weights, coreset_labels = stream_coreset(data, weights, labels, folds=folds)
    weighted_coreset = coreset * np.sqrt(coreset_weights[:, np.newaxis])
    weighted_coreset_labels = (coreset_labels * np.sqrt(coreset_weights[:, np.newaxis])).ravel()

    if solver in ['lasso', 'elastic']:
        const = np.sqrt(coreset.shape[0] / data.shape[0])
        clf.fit(const * weighted_coreset, const * weighted_coreset_labels)
    else:
        clf.fit(weighted_coreset, weighted_coreset_labels)
    time_end = time.time()

    return time_end - time_start, clf


def get_new_clf(solver, folds=3, alphas=100):
    kf=KFold(n_splits=folds,shuffle=False)
    if "linear" == solver:
        clf = linear_model.LinearRegression(fit_intercept=False)
    if "ridge" == solver:
        alphas =  np.arange(1/alphas, 10+ 1/alphas, 10/alphas)
        clf = linear_model.RidgeCV(alphas=alphas, fit_intercept=False, cv=kf)
    elif "lasso" == solver:
        clf=linear_model.LassoCV(n_alphas=alphas, fit_intercept=False, cv=kf)
    elif "elastic" == solver:
        clf = linear_model.ElasticNetCV(n_alphas=alphas, fit_intercept=False, cv=kf)
    return clf


def main():
    n = 240000
    d = 3
    data_range = 100
    num_of_alphas = 300
    folds = 3
    data = np.floor(np.random.rand(n, d) * data_range)
    labels = np.floor(np.random.rand(n, 1) * data_range)
    weights = np.ones(n)

    for solver in ["lasso", "ridge", "elastic"]:
        #########RIDGE REGRESSION#############
        clf = get_new_clf(solver, folds=folds, alphas=num_of_alphas)
        time_coreset, clf_coreset = coreset_train_model(data, labels, weights, clf, folds=folds, solver=solver)
        score_coreset = test_model(data, labels, weights, clf)

        clf = get_new_clf(solver, folds=folds, alphas=num_of_alphas)
        time_real, clf_real = train_model(data, labels, weights, clf)
        score_real = test_model(data, labels, weights, clf)

        print (" solver: {}\n number_of_alphas: {}, \nscore_diff = {}\n---->coef diff = {}\n---->coreset_time = {}\n---->data time = {}".format(
            solver,
            num_of_alphas,
            np.abs(score_coreset - score_real),
            np.sum(np.abs(clf_real.coef_ - clf_coreset.coef_)),
            time_coreset,
            time_real))
        ############################################




if __name__ == '__main__':
    main()