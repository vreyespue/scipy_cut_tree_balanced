import numpy as np
import pytest
import scipy
from scipy_cut_tree_balanced import cut_tree_balanced
from numpy.testing import assert_equal, assert_
from pytest import raises as assert_raises
from scipy.stats import gamma


def test_cut_tree_balanced():
    # Tests cut_tree_balanced(Z, max_cluster_size) on a random unbalanced data
    # set. There should be exactly one resulting cluster_id and cluster_level
    # for each of the input observations (i.e. input samples). Further, all
    # resulting clusters should contain a number of samples
    # "<= max_cluster_size". As a corollary, the highest cluster_id should be
    # ">= ((nobs // max_cluster_size) - 1)".
    nobs = 100
    np.random.seed(14)
    X = gamma.rvs(0.1, size=nobs * 4).reshape((nobs, 4))
    Z = scipy.cluster.hierarchy.ward(X)
    max_cluster_size = 10
    [cluster_id, cluster_level] = cut_tree_balanced(
        Z, max_cluster_size=max_cluster_size
    )
    uniq_id, uniq_count = np.unique(cluster_id, return_counts=True)
    assert_equal(cluster_id.shape, (nobs,))
    assert_equal(cluster_level.shape, (nobs,))
    assert_(all(elem <= max_cluster_size for elem in uniq_count))
    assert_(uniq_id.max() >= ((nobs // max_cluster_size) - 1))

    # Tests cut_tree_balanced(Z, max_cluster_size) on invalid input
    # "max_cluster_size = 0". The method should raise a ValueError.
    assert_raises(ValueError, cut_tree_balanced, Z, 0)

    # Tests cut_tree_balanced(Z, max_cluster_size) on invalid input
    # when the linkage matrix is empty (edge invalid case).
    Z = np.array([])
    assert_raises(ValueError, cut_tree_balanced, Z, max_cluster_size=max_cluster_size)

    # Tests cut_tree_balanced(Z, max_cluster_size) on invalid input
    # when the linkage matrix only has one element (edge invalid case).
    Z = np.array([[0.0]])
    assert_raises(ValueError, cut_tree_balanced, Z, max_cluster_size=max_cluster_size)

    # Tests cut_tree_balanced(Z, max_cluster_size) on hardcoded numbers for a
    # very small linkage tree (2 smples, edge valid case).
    X = np.array([[1], [2]])
    Z = scipy.cluster.hierarchy.ward(X)
    max_cluster_size = 4
    [cluster_id, cluster_level] = cut_tree_balanced(
        Z, max_cluster_size=max_cluster_size
    )
    assert_equal(cluster_id, [0, 0])
    assert_equal(cluster_level[0], [0])
    assert_equal(cluster_level[1], [0])

    # Tests cut_tree_balanced(Z, max_cluster_size) on hardcoded numbers for a
    # small linkage tree (8 smples). Since the data is unbalanced, there is a
    # bigger cluster containing 4 samples, and 2 clusters with 2 samples.
    # Note however that the size of all clusters is "<= max_cluster_size".
    X = np.array([[1], [2], [10], [11], [15], [16], [18], [19]])
    Z = scipy.cluster.hierarchy.ward(X)
    max_cluster_size = 4
    [cluster_id, cluster_level] = cut_tree_balanced(
        Z, max_cluster_size=max_cluster_size
    )
    assert_equal(cluster_id, [0, 0, 1, 1, 2, 2, 2, 2])
    assert_equal(cluster_level[0], [0, 1])
    assert_equal(cluster_level[1], [0, 1])
    assert_equal(cluster_level[2], [0, 0, 1])
    assert_equal(cluster_level[3], [0, 0, 1])
    assert_equal(cluster_level[4], [0, 0, 0])
    assert_equal(cluster_level[5], [0, 0, 0])
    assert_equal(cluster_level[6], [0, 0, 0])
    assert_equal(cluster_level[7], [0, 0, 0])

    # Tests cut_tree_balanced(Z, max_cluster_size) on the same hardcoded
    # numbers but a smaller max_cluster_size (= 2). Note that in this case the
    # biggest cluster is splitted into two smaller, deeper clusters with 2
    # samples each.
    max_cluster_size = 2
    [cluster_id, cluster_level] = cut_tree_balanced(
        Z, max_cluster_size=max_cluster_size
    )
    assert_equal(cluster_id, [0, 0, 1, 1, 2, 2, 3, 3])
    assert_equal(cluster_level[0], [0, 1])
    assert_equal(cluster_level[1], [0, 1])
    assert_equal(cluster_level[2], [0, 0, 1])
    assert_equal(cluster_level[3], [0, 0, 1])
    assert_equal(cluster_level[4], [0, 0, 0, 0])
    assert_equal(cluster_level[5], [0, 0, 0, 0])
    assert_equal(cluster_level[6], [0, 0, 0, 1])
    assert_equal(cluster_level[7], [0, 0, 0, 1])
