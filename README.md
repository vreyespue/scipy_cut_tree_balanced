# Balanced Cut Tree Method for Ward Hierarchical Clustering

This repo contains a small Python function that performs a balanced clustering by using the linkage matrix from a Ward histogram. It builds upon the SciPy and NumPy libraries.

The initial problem was the following: if you perform a standard cut on a tree (i.e. the result from a hierarchical clustering), probably you will end up having a few big clusters (where the number of data samples is high), and many small clusters (each containing very few data samples). Thus, the resulting clustering is unbalanced, i.e. it contains clusters of very variable size.

The proposed function looks recursively along the hierarchical tree, from the root (single cluster gathering all the samples) to the leaves (i.e. the clusters with only one sample), retrieving the biggest possible clusters containing a number of samples lower than a given maximum. In this way, if a cluster at a specific tree level contains a number of samples higher than the given maximum, it is ignored and its offspring (smaller) sub-clusters are taken into consideration. If the cluster contains a number of samples lower than the given maximum, it is taken as result and its offspring sub-clusters not further processed.

Since all output clusters contain no more than a given maximum number of samples, the resulting clustering is considered to be more balanced than a standard tree cut. Note however that the number of samples per cluster might still have a considerable variability, since the splitting of a big cluster can result in sub-clusters with very variable number of samples. This variability should be smaller as the given maximum of samples per cluster is closer to 1 (being the variability equal to 0 when the maximum is at its limit, i.e. 1).

The function returns two results: 
1. List of integers containing for each input sample its corresponding cluster id. The cluster id is an integer which is higher for deeper tree levels.
2. List of strings containing for each input sample its corresponding cluster tree level, i.e. a string of '0's and '1's separated by '.' Note that the cluster level is longer for deeper tree levels, being 0 the root cluster, 0.0 and 0.1 its offspring, and so on.

# Dependencies and Example Script

Before running the example script, please ensure you have installed the scipy and numpy packages in your Python environment.

In order to run the example script you can use the following command.

```
$ python3 ward_cut_tree_balanced.py
```

By running the example script you should run commands and get printed outputs similar to the following.

First, a numpy array of 1000 rows x 4 columns is randomly generated using a gamma distribution. Note that we perform such a random sampling from a gamma distribution so that the resulting standard clustering is unbalanced (see below). 

```
    np.random.seed(0)
    X = gamma.rvs(0.1, size=4000).reshape((1000,4))
```

In order to check the validity of the input data, the type, shape and the first 10 rows are printed.

```
Type of the input data sample: <class 'numpy.ndarray'>
Shape of the input data sample: (1000, 4)
First 10 rows of the input data:
[[2.47883654e-03 6.33094538e-03 1.86260732e-04 2.57422326e-04]
 [1.01239570e+00 9.67706807e-02 3.49803426e-03 3.27181353e-12]
 [1.14147734e-17 8.14087142e-02 1.69626003e+00 4.38049146e-04]
 [5.35683767e-10 3.66502184e-09 1.49776236e-03 1.67946048e-06]
 [3.90013783e-04 5.48556892e-18 7.38232987e-03 6.47944035e-01]
 [3.60648715e-05 2.73060529e-02 1.73675725e-02 1.69862447e-07]
 [9.75005924e-06 3.63285055e-03 1.58088814e-07 1.41205691e-02]
 [4.86132311e-04 1.03071920e-08 1.48326873e-02 8.61877379e-08]
 [1.54769645e+00 1.56690862e+00 3.27266064e-06 5.18768523e-06]]
```

Next, the linkage matrix is computed by using the ward method, and a standard tree cut is performed (with a specific number of output clusters = 20). 

```
    Z = ward(X)
    standard_cut_cluster_id = cut_tree(Z, n_clusters=[20])
```

As shown below, the output is a numpy array of 1000 elements, assigning one cluster ID to each input vector (of 4 dimensions, see above). Note that the ID of the resulting clusters go from 0 to 19 in this case. The resulting clustering is unbalanced, i.e. containing a big cluster (where the number of data samples is 506), and many small clusters (each containing very few data samples, one of them containing a single data sample). As result, the range of cluster sizes goes from 1 to 506, showing a standard deviation of 108.71 data samples.

```
Type of the standard clustering result: <class 'numpy.ndarray'>
Shape of the standard clustering result (one cluster id per data sample): (1000, 1)
First 10 rows of the standard clustering result (one cluster id per sample):
[0 1 2 0 3 0 0 0 4 0] ...
Total number of resulting clusters = 20
For each resulting cluster: Cluster ID
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
For each resulting cluster: Count of data samples
[506  18  15  84   2  80  40  54  31 100   8   3  18  19   5   5   3   5   3   1]
Count of data samples per cluster: mean = 50, max = 506, min = 1, std = 108.71
```

A more balanced clustering is then attempted by using the balanced ward tree method, in which the maximum number of data samples within each cluster is set to 100. 

```
    [balanced_cut_cluster_id, balanced_cut_cluster_level] = ward_cut_tree_balanced(Z, 100, verbose=False)
```

We get two results from the new function: (1) a list of integers containing for each input sample its corresponding cluster id, and (2) a list of strings containing for each input sample its corresponding cluster tree level (see above section for further information). Note that the ID of the resulting clusters go from 1 to 20 in this case, i.e. the number of resulting clusters (20) is identical to the previous one. Importantly, the resulting clustering is more balanced than the standard one (for an equal number of resulting clusters), since the range of cluster sizes goes from 7 to 100, showing a standard deviation of 29.08 data samples.

```
Type of the balanced clustering result: <class 'numpy.ndarray'>
Shape of the balanced clustering result (one cluster id per data sample): (1000,)
First 10 rows of the balanced clustering result (one cluster id per sample):
[15  3  1 18  6 11 14 12  3 18] ...

Type of the balanced clustering result (level): <class 'numpy.ndarray'>
Shape of the balanced clustering result (level) (one string per data sample): (1000,)
First 10 rows of the balanced clustering result (level) (one string per sample):
['0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0' '0.0.0.1' '0.1'
 '0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.1.0.0.0' '0.0.0.0.0.0.1'
 '0.0.0.0.0.0.0.0.0.0.0.1' '0.0.0.0.0.0.0.0.0.0.0.0.0.0.1'
 '0.0.0.0.0.0.0.0.0.0.0.0.1' '0.0.0.1'
 '0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.1.0.0.0'] ...

Total number of resulting clusters = 20
For each resulting cluster: Cluster ID
[ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20]
For each resulting cluster: Count of data samples
[ 39  93  69  29  80  84 100  65  67  63  43  47  34  23  27  15  13   7  89  13]
Count of data samples per cluster: mean = 50, max = 100, min = 7, std = 29.08
```

In conclusion, here we describe and implement a method which generates (for a similar number of resulting clusters) a more balanced outcome, i.e. building clusters of less variable size.