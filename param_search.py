import numpy as np
import pandas as pd
from scipy_cut_tree_balanced import cut_tree_balanced
from scipy.cluster.hierarchy import linkage
from sklearn import datasets, metrics

bool_print_info = True
max_silh_sample_size = 1000  # max num of samples to compute the silhouette metric


def comp_silh_for_max_cluster_sizes(X, linkage_matrix, list_max_cluster_sizes):
    """
    Given a input data X, a linkage matrix and a list of max_cluster_sizes,
    return the average silhouette metric (computed along all resulting clusters)
    for each value of max_cluster_size.

    Parameters
    ----------
    X : ndarray
        Input data (each row contains a data sample of n dim)
    list_max_cluster_sizes : list
        List of values for max_cluster_size, which denotes the maximum
        number of data samples contained within the resulting clusters.
        Thus, all resulting clusters will contain a number of data samples
        ``<= max_cluster_size``. Must be >= 1.

    Returns
    -------
    df_output : dataframe
        Dataframe containing the results related to each specific value of
        max_cluster_size, including (a) the resulting number of clusters,
        (b) the average silhouette metric computed along all resulting classes,
        (c) a dict containing for each cluster id its count, i.e. the num of
        samples contained within the cluster.
    """

    # Prepare a dataframe to contain the results
    df_output = pd.DataFrame(
        columns=[
            "max_cluster_size",
            "n_clusters",
            "avg_silhouette",
            "cluster_values_counts",
        ]
    )

    for max_cluster_size in list_max_cluster_sizes:

        # Perform a balanced cut tree of the linkage matrix
        [cluster_indices, _] = cut_tree_balanced(linkage_matrix, max_cluster_size)
        cluster_values, cluster_counts = np.unique(cluster_indices, return_counts=True)
        n_clusters = cluster_values.shape[0]

        # Sample data for computing the silhouette metric
        input_data_x_sample = None
        cluster_indices_sample = None
        for curr_cluster_id in range(0, n_clusters, 1):
            list_occurrences = [
                i for i, x in enumerate(cluster_indices) if x == curr_cluster_id
            ]
            if input_data_x_sample is None:
                input_data_x_sample = X[list_occurrences[0:max_silh_sample_size]]
            else:
                input_data_x_sample = np.vstack(
                    (
                        input_data_x_sample,
                        X[list_occurrences[0:max_silh_sample_size]],
                    )
                )
            if cluster_indices_sample is None:
                cluster_indices_sample = np.array(cluster_indices)[
                    list_occurrences[0:max_silh_sample_size]
                ]
            else:
                cluster_indices_sample = np.hstack(
                    (
                        cluster_indices_sample,
                        np.array(cluster_indices)[
                            list_occurrences[0:max_silh_sample_size]
                        ],
                    )
                )

        # Compute mean silhouette for each class and the average across all classes
        try:
            silh_array = metrics.silhouette_samples(
                input_data_x_sample,
                np.asarray(cluster_indices_sample),
                metric="euclidean",
            )
            np_silh_samples = np.column_stack(
                (cluster_indices_sample, silh_array.tolist())
            )
            df_silh_samples = pd.DataFrame(
                data=np_silh_samples[0:, 0:], columns=["cluster_id", "silhouette"]
            )
            df_silh_mean_per_class = df_silh_samples.groupby(
                ["cluster_id"]
            ).mean()  # .sort_values(by='cluster_id')
            df_silh_mean_per_class.reset_index(level=0, inplace=True)
            df_silh_mean_per_class.sort_values(by="cluster_id")
            avg_silhouette = df_silh_mean_per_class["silhouette"].mean()
        except ValueError:
            avg_silhouette = np.nan

        # Append the results to the output dataframe
        df_output = df_output.append(
            pd.Series(
                [
                    max_cluster_size,
                    n_clusters,
                    avg_silhouette,
                    str(dict(zip(cluster_values, cluster_counts))),
                ],
                index=df_output.columns,
            ),
            ignore_index=True,
        )

    return df_output


if __name__ == "__main__":

    print("\ndataset iris\n")
    data_bunch = datasets.load_iris()
    X = data_bunch.data[:]
    Y = data_bunch.target
    linkage_matrix = linkage(X, "ward")
    df_output = comp_silh_for_max_cluster_sizes(X, linkage_matrix, range(150, 0, -25))
    if bool_print_info:
        print("input data (X) shape: %s" % str(X.shape))
        print(str(X[0:3]))
        print("...\n")
        unique, counts = np.unique(Y, return_counts=True)
        print("target class (Y) values and counts: %s" % str(dict(zip(unique, counts))))
        print("")
        print(df_output)
        print("")

    print("\ndataset digits\n")
    data_bunch = datasets.load_digits()
    X = data_bunch.data[:]
    Y = data_bunch.target
    linkage_matrix = linkage(X, "ward")
    df_output = comp_silh_for_max_cluster_sizes(X, linkage_matrix, range(900, 0, -200))
    if bool_print_info:
        print("input data (X) shape: %s" % str(X.shape))
        print(str(X[0:3]))
        print("...\n")
        unique, counts = np.unique(Y, return_counts=True)
        print("target class (Y) values and counts: %s" % str(dict(zip(unique, counts))))
        print("")
        print(df_output)
        print("")
