from scipy_cut_tree_balanced import cut_tree_balanced
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.stats import gamma
import numpy as np


if __name__ == "__main__":

    # Initialize the random seed
    np.random.seed(14)
    # Create a input matrix containing 100 data samples with 4 dimensions
    # Note: random sample from gamma distribution in order to obtain an unbalanced clustering (see below)
    X = gamma.rvs(0.1, size=400).reshape((100, 4))
    print("")
    print("Type of the input data sample: %s" % type(X))
    print("Shape of the input data sample: %s" % str(X.shape))
    print("First 10 rows of the input data:")
    print(X[0:9])
    print("")

    # Compute the linkage matrix using the scipy linkage() method
    Z = linkage(X, "ward")

    # Perform standard clustering by cutting the tree at a certain level (where the nr of clusters is set to 20)
    standard_cut_cluster_id = cut_tree(Z, n_clusters=[20])
    print("Type of the standard clustering result: %s" % type(standard_cut_cluster_id))
    print(
        "Shape of the standard clustering result (one cluster id per data sample): %s"
        % str(standard_cut_cluster_id.shape)
    )
    print(
        "First 10 rows of the standard clustering result (one cluster id per sample):"
    )
    print(str(standard_cut_cluster_id[0:10].reshape(10)) + " ...")
    standard_cluster_values, standard_cluster_counts = np.unique(
        standard_cut_cluster_id, return_counts=True
    )
    print("Total number of resulting clusters = %s" % standard_cluster_values.shape[0])
    print("For each resulting cluster: Cluster ID")
    print(standard_cluster_values)
    print("For each resulting cluster: Count of data samples")
    print(standard_cluster_counts)
    print(
        "Count of data samples per cluster: mean = %d, max = %d, min = %d, std = %.2f"
        % (
            np.mean(standard_cluster_counts),
            np.max(standard_cluster_counts),
            np.min(standard_cluster_counts),
            np.std(standard_cluster_counts),
        )
    )
    print("")

    # Perform a balanced cut tree of the linkage matrix
    [balanced_cut_cluster_id, balanced_cut_cluster_level] = cut_tree_balanced(Z, 10)
    print(
        "Type of the balanced clustering result (id): %s"
        % type(balanced_cut_cluster_id)
    )
    print(
        "Shape of the balanced clustering result (one cluster id per data sample): %s"
        % str(balanced_cut_cluster_id.shape)
    )
    print(
        "First 10 rows of the balanced clustering result (one cluster id per sample):"
    )
    print(str(balanced_cut_cluster_id[0:10]) + " ...")
    print("")

    print(
        "Type of the balanced clustering result (level): %s"
        % type(balanced_cut_cluster_level)
    )
    print(
        "Shape of the balanced clustering result (level) (one array per data sample): %s"
        % str(balanced_cut_cluster_level.shape)
    )
    print(
        "First 10 rows of the balanced clustering result (level) (one array per sample):"
    )
    print(str(balanced_cut_cluster_level[0:10]) + " ...")
    print("")

    balanced_cluster_values, balanced_cluster_counts = np.unique(
        balanced_cut_cluster_id, return_counts=True
    )
    print("Total number of resulting clusters = %s" % balanced_cluster_values.shape[0])
    print("For each resulting cluster: Cluster ID")
    print(balanced_cluster_values)
    print("For each resulting cluster: Count of data samples")
    print(balanced_cluster_counts)
    print(
        "Count of data samples per cluster: mean = %d, max = %d, min = %d, std = %.2f"
        % (
            np.mean(balanced_cluster_counts),
            np.max(balanced_cluster_counts),
            np.min(balanced_cluster_counts),
            np.std(balanced_cluster_counts),
        )
    )
    print("")
