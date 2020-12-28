from scipy.cluster.hierarchy import cut_tree
import numpy as np


def cut_tree_balanced(Z, max_cluster_size):
    """
    Given a linkage matrix Z and max cluster size, return a balanced cut tree.

    The function looks recursively along the hierarchical tree, from the root
    (single cluster gathering all the samples) to the leaves (i.e. the clusters
    with only one sample), retrieving the biggest possible clusters containing
    a number of samples lower than a given maximum. If a cluster at a specific
    tree level contains a number of samples higher than the given maximum, it
    is ignored and its offspring (smaller) sub-clusters are taken into
    consideration. If the cluster contains a number of samples lower than the
    given maximum, it is taken as result and its offspring sub-clusters not
    further processed.

    Parameters
    ----------
    Z : ndarray
        The linkage matrix resulting from calling `ward` or `linkage`. I.e. it
        contains the hierarchical clustering encoded as a linkage matrix.
    max_cluster_size : int
        Maximum number of data samples contained within the resulting clusters.
        Thus, all resulting clusters will contain a number of data samples
        ``<= max_cluster_size``. Must be >= 1.

    Returns
    -------
    cluster_id : ndarray
        One-dimensional array of integers containing for each input sample its
        corresponding cluster id. The cluster id is an integer which is higher
        for deeper tree levels.
    cluster_level : ndarray
        One-dimensional array of integer arrays containing for each input
        sample its corresponding cluster tree level, i.e. a sequence of
        0's and 1's. Note that the cluster level is longer for deeper tree
        levels, being [0] the root cluster, [0, 0] and [0, 1] its offspring,
        and so on. Also note that in each cluster splitting, the label 0
        denotes the bigger cluster, while the label 1 denotes the smallest.

    See Also
    --------
    cut_tree

    Notes
    -----
    There are several implemented methods following the same idea, i.e.
    performing a tree cut in which the resulting clusters are at different tree
    levels, but using more elaborated algorithms (in which the threshold of
    ``max_cluster_size`` is dynamically computed). The CRAN R package
    dynamicTreeCut (github.com/cran/dynamicTreeCut) implements novel
    dynamic branch cutting methods for detecting clusters in a dendrogram
    depending on their shape. Further, MLCut (github.com/than8/MLCut)
    provides interactive methods to cut tree branches at multiple levels.
    Note that in the present method, the ``max_cluster_size`` threshold is a
    fixed value given as input.

    Further, note that this algorithm uses :math:`O(n^2)` memory, i.e. the same
    as `cut_tree` because a full cut tree of the linkage matrix is performed
    as the beginning. This data structure ``full_cut`` is used in order to
    perform the successive computations.

    Graphical examples of this algorithm can be found at the original repo
    describing this method (github.com/vreyespue/cut_tree_balanced).

    Examples
    --------
    >>> from scipy.cluster import hierarchy
    >>> from scipy import stats

    Initialize the random seed.

    >>> np.random.seed(14)

    Create a input matrix containing 100 data samples with 4 dimensions.
    Note: using `gamma` in order to generate an unbalanced distribution.
    If a regular ``cut_tree()`` would be performed, one big and many small
    clusters would be obtained.

    >>> X = stats.gamma.rvs(0.1, size=400).reshape((100, 4))

    Compute the linkage matrix using the scipy ward() or linkage() method:

    >>> Z = hierarchy.ward(X)

    Perform a balanced cut tree of the linkage matrix:

    >>> cluster_id, cluster_level = hierarchy.cut_tree_balanced(
    ...                                              Z, max_cluster_size=10)
    >>> cluster_id[:10]
    array([18,  3,  9, 11, 19, 11, 13,  8, 14,  1])
    >>> cluster_level[:10]
    array([array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
           array([0, 0, 0, 1]),
           array([0, 0, 0, 0, 0, 0, 0, 0, 1]),
           array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
           array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
           array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
           array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
           array([0, 0, 0, 0, 0, 0, 0, 1]),
           array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
           array([0, 0, 1, 0])], dtype=object)

    Note that clusters with more similar values for ``cluster_level`` denote
    clusters with less distance in between, thus representing vectors which are
    closer in the multidimensional space. This information contained within
    ``cluster_level`` is not usually present in ``cluster_id``.

    """
    # Assert that the input max_cluster_size is >= 1
    if not max_cluster_size >= 1:
        raise ValueError(
            "max_cluster_size should be >= 1, is: {}".format(max_cluster_size)
        )

    # Perform a full cut tree of the linkage matrix
    full_cut = cut_tree(Z)

    # Initialize the variable containing the current cluster id (it will be
    # higher for each newly found valid cluster)
    last_cluster_id = 0

    # Initialize the resulting cluster id vector (containing for each row in
    # input_data_x_sample its corresponding cluster id)
    ndim = full_cut.shape[1]
    cluster_id = np.zeros(ndim, dtype=int)

    # Initialize the resulting cluster level vector (containing for each data
    # sample its corresponding cluster tree level)
    cluster_level = np.empty((ndim,), dtype=object)
    for i in range(ndim):
        cluster_level[i] = np.array([0], int)

    # Scan the full cut matrix from the last column (root tree level) to the
    # first column (leaves tree level)
    for icol in range(ndim - 1, -1, -1):
        # Get a list of unique group ids and their count within the current
        # tree level
        values, counts = np.unique(full_cut[:, icol], return_counts=True)

        # Stop if all samples have been already selected (i.e. if all data
        # samples have been already clustered)
        if (values.size == 1) and (values[0] == -1):
            break

        # For each group id within the current tree level
        for ival in range(values.size):
            # If it is a valid group id (i.e. not yet marked with -1)
            # Note: data samples which were alredy included in a valid
            # cluster id are marked with the group id -1 (see below)
            if values[ival] >= 0:
                # Select the current group id
                selected_curr_value = values[ival]

                # Look for the vector positions (related to rows in
                # input_data_x_sample) belonging to the current group id
                selected_curr_elems = np.where(full_cut[:, icol] == selected_curr_value)

                # Major step #1: Populate the resulting vector of cluster
                # levels for each data sample, if we are not at the root
                if icol < (ndim - 1):
                    # Get the ancestor values and element positions
                    selected_ancestor_value = full_cut[
                        selected_curr_elems[0][0], icol + 1
                    ]
                    selected_ancestor_elems = np.where(
                        full_cut[:, icol + 1] == selected_ancestor_value
                    )

                    # Compute the values and counts of the offspring and sort
                    # them by their count (so that the biggest cluster gets the
                    # offspring_elem_label = 0, see below)
                    offspring_values, offspring_counts = np.unique(
                        full_cut[selected_ancestor_elems, icol], return_counts=True
                    )
                    count_sort_ind = np.argsort(-offspring_counts)
                    offspring_values = offspring_values[count_sort_ind]
                    offspring_counts = offspring_counts[count_sort_ind]

                    # If the size of the offspring is > 1
                    if offspring_values.shape[0] > 1:
                        # Select the label of the current value (i.e. 0 or 1)
                        # and append it to the cluster level
                        offspring_elem_label = np.where(
                            offspring_values == selected_curr_value
                        )[0][0]
                        for i in selected_curr_elems[0]:
                            cluster_level[i] = np.hstack(
                                (cluster_level[i], offspring_elem_label)
                            )

                # Major step #2: Populate the resulting vector of cluster ids
                # for each data sample, and mark them as clustered (-1)
                # If the number of elements is below max_cluster_size
                if counts[ival] <= max_cluster_size:
                    # Relate vector positions to the current cluster id
                    cluster_id[selected_curr_elems] = last_cluster_id

                    # Delete these vector positions at lower tree levels for
                    # further processing (i.e. mark as clustered)
                    full_cut[selected_curr_elems, 0:icol] = -1

                    # Update the cluster id
                    last_cluster_id += 1

    # Return the resulting clustering array (containing for each row in
    # input_data_x_sample its corresponding cluster id)
    return cluster_id, cluster_level
