"""
Creates a weighted sample graph with nodes as cells with weights calculated
through a radial basis kernel. Then calculates the modularity of this weighted
sample graph.

Credit: Modularity and the equation for calculating modularity was proposed by
Newman and Girvan. See the citation in our paper for details.

@author: Leo Zhengyang Dong
@contact: leozdong@stanford.edu
@date: 10/16/2019
"""

# import packages
import torch
from torch import Tensor
import math

# define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def make_aff_matrix(features, gamma):
    """ Creates a weighted sample graph represented as an affinity matrix.
    The closer the two samples are, the higher affinity score they will have.
    Affinity score is in the range between 0 to 1.

    Parameters
    ----------
    features : tensor, shape (n_samples, n_features)
        Each row is a sample and each column is a feature.

    Returns
    -------
    aff_matrix : tensor, shape (n_samples, n_samples)
        aff_matrix[i, j] is the affinity score of sample i and j measured
        by the features that represent these two samples.
        aff_matrix is a symmetric matrix where the diagonal is all 1.
    """
    # start = time.time()
    dist_matrix = torch.norm(features[:, None] - features, dim=2, p=2)
    # print("dist_matrix time:", time.time() - start)

    aff_matrix = (torch.exp(-gamma * dist_matrix) + 1e-8).to(device)

    # subtract to make diagonal all zeros
    diag = (torch.diag(torch.diagonal(aff_matrix, 0))).to(device)  # extract diagonal entries as a diagonal matrix

    aff_matrix = aff_matrix - diag

    return aff_matrix


def calc_modularity(features, labels, gamma):
    """ Calculate the weighted network modularity measure of the generated
    weighted sample graph.

    Parameters
    ----------
    features : tensor, shape (n_samples, n_features)
        Each row is a sample and each column is a feature.
    labels : tensor, shape (n_samples, 1)
        Records the class label of each sample.

    Returns
    -------
    modularity : numeric
        network modularity measure of the weighted sample graph.
    """

    # 1. Make identity vector s:
    s_vec = 2 * labels - torch.ones((labels.size()[0], 1), device=device)
    s_vec = s_vec.to(device)
    # 2. Make affinity matrix A:
    A_mat = make_aff_matrix(features, gamma)
    # 3. Make matrix K:
    k_vec = torch.mm(A_mat, torch.ones((A_mat.size()[1], 1), device=device))
    k_vec = k_vec.to(device)
    K_mat = torch.mm(k_vec, torch.t(k_vec))
    K_mat = K_mat.to(device)
    A_sum = torch.sum(A_mat).to(device)  # A_sum is `2m` in the original definition

    # 4. Make modularity matrix B:
    B_mat = A_mat - (1/A_sum) * K_mat
    B_mat = B_mat.to(device)

    # 5. Calculate modularity Q:
    # q = (1 / (2 * A_sum)) * torch.mm(torch.mm(torch.t(s_vec), B_mat), s_vec)
    q = (1 / A_sum) * torch.mm(torch.mm(torch.t(s_vec), B_mat), s_vec)
    q = torch.sum(q)
    q = q.to(device)

    return q


def calc_modularity_mult(partitions, gamma):
    """ Calculate modularity for multiple sets.
    This is used for calculating modularity of more than two classes when the procedure
    is simply repeatedly dividing into binary classes and calculating the modularity in the binary case.

    Parameters
    ----------
    partitions : list of lists [set1, set2, set3, ...]
        set1 = [features of set1, labels of set1]
    """
    q = 0
    for features, labels in partitions:
        features = features.to(device)
        labels = labels.to(device)
        delta_q = calc_modularity(features, labels, gamma)
        q += delta_q
    q /= len(partitions)
    q = q.to(device)
    return q
