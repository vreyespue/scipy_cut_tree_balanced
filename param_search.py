import numpy as np
import pandas as pd
from scipy_cut_tree_balanced import cut_tree_balanced
from scipy.cluster.hierarchy import linkage
from sklearn import datasets, metrics

const_silh_max_samples = 1000  # max num of samples to compute the silhouette metric
const_silh_distance = "euclidean"  # distance to compute the silhouette metric
const_linkage_method = "ward"  # method to compute the linkage matrix
const_bool_print_info = True  # boolean print info to stdout


def comp_avg_silh_metric(data_input, cluster_indices, silh_max_samples, silh_distance):
    """
    Given a input data matrix and an array of cluster indices, returns the
    average silhouette metric for that clustering result (computed across all clusters).

    Parameters
    ----------
    data_input : ndarray
        Data to be clustered (each row contains a n-dimensional data sample)
    cluster_indices : list
        List containing for each data point (each row in data input) its cluster id
    silh_max_samples: int
        Maximum number of samples to compute the silhouette metric (higher for
        more exact values at higher computing costs)
    silh_distance: string
        Metric to use when calculating distance between instances
        e.g. 'euclidean', 'manhattan', 'cosine'

    Returns
    -------
    avg_silhouette : float
        Silhouette metric averaged across all clusters
    """

    # Sample data for computing the silhouette metric
    input_data_x_sample = None
    cluster_indices_sample = None
    for curr_cluster_id in set(cluster_indices):
        list_occurrences = [
            i for i, x in enumerate(cluster_indices) if x == curr_cluster_id
        ]
        if input_data_x_sample is None:
            input_data_x_sample = data_input[list_occurrences[0:silh_max_samples]]
        else:
            input_data_x_sample = np.vstack(
                (
                    input_data_x_sample,
                    data_input[list_occurrences[0:silh_max_samples]],
                )
            )
        if cluster_indices_sample is None:
            cluster_indices_sample = np.array(cluster_indices)[
                list_occurrences[0:silh_max_samples]
            ]
        else:
            cluster_indices_sample = np.hstack(
                (
                    cluster_indices_sample,
                    np.array(cluster_indices)[list_occurrences[0:silh_max_samples]],
                )
            )

    # Compute mean silhouette for each class and the average across all classes
    try:
        silh_array = metrics.silhouette_samples(
            input_data_x_sample,
            np.asarray(cluster_indices_sample),
            metric=silh_distance,
        )
        np_silh_samples = np.column_stack((cluster_indices_sample, silh_array.tolist()))
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

    return avg_silhouette


def comp_silh_for_max_cluster_sizes(
    data_input, list_max_cluster_sizes, silh_max_samples, silh_distance, linkage_method
):
    """
    Given a input data matrix, a linkage matrix and a list of max_cluster_sizes,
    returns the average silhouette metric for each value of max_cluster_size
    (computed across all clusters). Note that the optimal cut is related
    to the maximum value for the silhouette metric.

    Parameters
    ----------
    data_input : ndarray
        Data to be clustered (each row contains a n-dimensional data sample)
    list_max_cluster_sizes : list
        List of values for max_cluster_size, which denotes the maximum
        number of data samples contained within the resulting clusters.
        Thus, all resulting clusters will contain a number of data samples
        ``<= max_cluster_size``. Must be >= 1.
    silh_max_samples: int
        Maximum number of samples to compute the silhouette metric (higher for
        more exact values at higher computing costs)
    silh_distance: string
        Metric to use when calculating distance between instances
        e.g. 'euclidean', 'manhattan', 'cosine'
    linkage_method: string
        Method to compute the linkage matrix, e.g. 'ward', 'weighted'

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

    linkage_matrix = linkage(data_input, linkage_method)

    for max_cluster_size in list_max_cluster_sizes:

        # Perform a balanced cut tree of the linkage matrix
        [cluster_indices, _] = cut_tree_balanced(linkage_matrix, max_cluster_size)
        cluster_values, cluster_counts = np.unique(cluster_indices, return_counts=True)
        n_clusters = cluster_values.shape[0]

        # Compute average silhouette metric for specific max_cluster_size
        avg_silhouette = comp_avg_silh_metric(
            data_input, cluster_indices, silh_max_samples, silh_distance
        )

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


def process_data_bunch_max_cluster_sizes(
    data_bunch,
    list_max_cluster_sizes,
    silh_max_samples,
    silh_distance,
    linkage_method,
    bool_print_info,
):
    """
    Given a input data bunch, and a list of max_cluster_sizes, returns the results
    (including the average silhouette metric) for each value of max_cluster_size
    (see function comp_silh_for_max_cluster_sizes for further details).

    Parameters
    ----------
    data_bunch : data bunch
        Input data as a result of datasets.load_*
    list_max_cluster_sizes : list
        List of values for max_cluster_size (see function
        comp_silh_for_max_cluster_sizes for further details)
    silh_max_samples: int
        Maximum number of samples to compute the silhouette metric (higher for
        more exact values at higher computing costs)
    silh_distance: string
        Metric to use when calculating distance between instances
        e.g. 'euclidean', 'manhattan', 'cosine'
    linkage_method: string
        Method to compute the linkage matrix, e.g. 'ward', 'weighted'
    bool_print_info: boolean
        Set if info should be printed to stdout

    Returns
    -------
    df_output : dataframe
        Dataframe containing the results related to each specific value of
        max_cluster_size (see function comp_silh_for_max_cluster_sizes for
        further details).
    """

    X = data_bunch.data[:]
    Y = data_bunch.target
    df_output = comp_silh_for_max_cluster_sizes(
        X, list_max_cluster_sizes, silh_max_samples, silh_distance, linkage_method
    )

    if bool_print_info:
        print("input data (X) shape: %s" % str(X.shape))
        print(str(X[0:3]))
        print("...\n")
        unique, counts = np.unique(Y, return_counts=True)
        print("target class (Y) values and counts: %s" % str(dict(zip(unique, counts))))
        print("")
        print(df_output)
        print("")

    return df_output


if __name__ == "__main__":

    if const_bool_print_info:
        print("\ndataset iris\n")
    df_output_iris = process_data_bunch_max_cluster_sizes(
        datasets.load_iris(),
        range(150, 0, -25),
        const_silh_max_samples,
        const_silh_distance,
        const_linkage_method,
        const_bool_print_info,
    )

    if const_bool_print_info:
        print("\ndataset digits\n")
    df_output_digits = process_data_bunch_max_cluster_sizes(
        datasets.load_digits(),
        range(900, 0, -200),
        const_silh_max_samples,
        const_silh_distance,
        const_linkage_method,
        const_bool_print_info,
    )
