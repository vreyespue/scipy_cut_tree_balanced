from scipy.cluster.hierarchy import cut_tree
import numpy as np


def cut_tree_balanced(linkage_matrix_Z, max_cluster_size, verbose=False):
    """This function performs a balanced cut tree of a SciPy linkage matrix built using any linkage method
    (e.g. 'ward'). It builds upon the SciPy and Numpy libraries.

    The function looks recursively along the hierarchical tree, from the root (single cluster gathering
    all the samples) to the leaves (i.e. the clusters with only one sample), retrieving the biggest
    possible clusters containing a number of samples lower than a given maximum. In this way, if a
    cluster at a specific tree level contains a number of samples higher than the given maximum, it is
    ignored and its offspring (smaller) sub-clusters are taken into consideration. If the cluster contains
    a number of samples lower than the given maximum, it is taken as result and its offspring sub-clusters
    not further processed.

    Input parameters:

       linkage_matrix_Z: linkage matrix resulting from calling the method scipy.cluster.hierarchy.ward()
          I.e. it contains the hierarchical clustering encoded as a linkage matrix.

       max_cluster_size: maximum number of data samples contained within the resulting clusters. Thus, all
          resulting clusters will contain a number of data samples <= max_cluster_size.
          Note that max_cluster_size must be >= 1.

       verbose: activates (True) / deactivates (False) some output print commands, which can be useful to
          test and understand the proposed tree cut method.

    Returns:

       vec_cluster_id: one-dimensional numpy array of integers containing for each input sample its corresponding
          cluster id. The cluster id is an integer which is higher for deeper tree levels.

       vec_last_cluster_level: one-dimensional numpy array of arrays containing for each input sample its
          corresponding cluster tree level, i.e. a sequence of 0s and 1s. Note that the cluster level is longer for
          deeper tree levels, being [0] the root cluster, [0, 0] and [0, 1] its offspring, and so on. Also note that
          in each cluster splitting, the label 0 denotes the bigger cluster, while the label 1 denotes the smallest.

    """
    try:
        # Assert that the input max_cluster_size is >= 1
        assert max_cluster_size >= 1

        # Perform a full cut tree of the linkage matrix, i.e. containing all tree levels
        full_cut = cut_tree(linkage_matrix_Z)
        if verbose:
            print("Interim full cut tree (square matrix)")
            print("Shape = " + str(full_cut.shape))
            print(full_cut)
            print("")

        # Initialize the vble containing the current cluster id (it will be higher for each newly
        # found valid cluster, i.e. for each found cluster with <= max_cluster_size data samples)
        last_cluster_id = 1

        # Initialize the resulting cluster id vector (containing for each row in input_data_x_sample
        # its corresponding cluster id)
        vec_cluster_id = np.zeros(full_cut.shape[1], dtype=int)

        # Initialize the resulting cluster level vector (containing for each data sample its
        # corresponding cluster tree level, i.e. a string of '0's and '1's separated by '.')
        vec_last_cluster_level = np.empty((full_cut.shape[1],), dtype=object)
        for i in range(full_cut.shape[1]):
            vec_last_cluster_level[i] = np.array([0], int)

        # Scan the full cut matrix from the last column (root tree level) to the first column (leaves tree level)
        if verbose:
            print(
                "Note about columns: within the full cut tree, the column "
                + str(full_cut.shape[1] - 1)
                + " represents the root, while 0 represent the leaves."
            )
            print(
                "We now scan the full cut tree from the root (column "
                + str(full_cut.shape[1] - 1)
                + ") "
                "to the leaves (column 0)."
            )
            print("")

        for curr_column in range(full_cut.shape[1] - 1, -1, -1):

            # Get a list of unique group ids and their count within the current tree level
            values, counts = np.unique(full_cut[:, curr_column], return_counts=True)

            # Stop if all samples have been already selected (i.e. if all data samples have been already clustered)
            if (values.size == 1) and (values[0] == -1):
                break

            # For each group id within the current tree level
            for curr_elem_pos in range(values.size):

                # If it is a valid group id (i.e. not yet marked as processed with -1) ...
                # Note: data samples which were alredy included in a valid cluster id (i.e. at a higher tree level)
                # are marked with the group id -1 (see below)
                if values[curr_elem_pos] >= 0:

                    # Select the current group id
                    selected_curr_value = values[curr_elem_pos]

                    # Look for the vector positions (related to rows in input_data_x_sample) belonging to
                    # the current group id
                    selected_curr_elems = np.where(
                        full_cut[:, curr_column] == selected_curr_value
                    )

                    # Major step #1: Populate the resulting vector of cluster levels for each data sample
                    # If we are not at the root level (i.e. single cluster gathering all the samples) ...
                    if curr_column < (full_cut.shape[1] - 1):
                        # Get the ancestor values and element positions
                        selected_ancestor_value = full_cut[
                            selected_curr_elems[0][0], curr_column + 1
                        ]
                        selected_ancestor_elems = np.where(
                            full_cut[:, curr_column + 1] == selected_ancestor_value
                        )

                        # Compute the values and counts of the offspring (i.e. curr_elem + brothers) and sort them
                        # by their count (so that the biggest cluster gets the offspring_elem_label = 0, see below)
                        offspring_values, offspring_counts = np.unique(
                            full_cut[selected_ancestor_elems, curr_column],
                            return_counts=True,
                        )
                        count_sort_ind = np.argsort(-offspring_counts)
                        offspring_values = offspring_values[count_sort_ind]
                        offspring_counts = offspring_counts[count_sort_ind]

                        # If the number of descendants is > 1 (i.e. if the curr_elem has at least one brother)
                        if offspring_values.shape[0] > 1:
                            # Select the position of the current value (i.e. 0 or 1) and append it to the cluster level
                            offspring_elem_label = np.where(
                                offspring_values == selected_curr_value
                            )[0][0]
                            for i in selected_curr_elems[0]:
                                vec_last_cluster_level[i] = np.hstack(
                                    (vec_last_cluster_level[i], offspring_elem_label)
                                )

                    # Major step #2: Populate the resulting vector of cluster ids for each data sample,
                    # and mark them as already clustered (-1)
                    # If the number of elements is below max_cluster_size ...
                    if counts[curr_elem_pos] <= max_cluster_size:

                        if verbose:
                            print(
                                "Current column in full cut tree = " + str(curr_column)
                            )
                            print("list_group_ids:     " + str(values))
                            print("list_count_samples: " + str(counts))
                            print(
                                "selected_curr_value: "
                                + str(selected_curr_value)
                                + ", count_samples = "
                                + str(counts[curr_elem_pos])
                                + ", marked as result"
                            )
                            print("")

                        # Relate these vector positions to the current cluster id
                        vec_cluster_id[selected_curr_elems] = last_cluster_id

                        # Delete these vector positions at the lower tree levels for further processing
                        # (i.e. mark these elements as already clustered)
                        full_cut[selected_curr_elems, 0:curr_column] = -1

                        # Update the cluster id
                        last_cluster_id = last_cluster_id + 1

        # Return the resulting clustering array (containing for each row in input_data_x_sample its
        # corresponding cluster id) and the clustering level
        return vec_cluster_id, vec_last_cluster_level

    except AssertionError:
        print("Please use a max_cluster_size >= 1")
