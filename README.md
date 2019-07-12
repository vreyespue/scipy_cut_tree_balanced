# Balanced Cut Tree Method for Ward Hierarchical Clustering

This repo contains a small Python function that performs a balanced clustering by using the linkage matrix from a Ward histogram. It builds upon the scipy and numpy libraries.

The initial problem was the following: if you perform a standard cut on a tree (i.e. the result from a hierarchical clustering), probably you will end up having a few big clusters (where the number of data samples is high), and many small clusters (containing each very few data samples). Thus, the resulting clustering is unbalanced, i.e. it contains clusters of very variable size.

The proposed function looks recursively along the hierarchical tree, from the root (single cluster gathering all the samples) to the leaves (i.e. the clusters with only one sample), retrieving the biggest possible clusters containing a number of samples lower than a given maximum. In this way, if a cluster at a specific tree level contains a number of samples higher than the given maximum, it is ignored and its offspring (smaller) sub-clusters are taken into consideration. If the cluster contains a number of samples lower than the given maximum, it is taken as result and its offspring sub-clusters not further processed.

Since all output clusters contain no more than a given maximum number of samples, the resulting clustering is considered to be more balanced than a standard tree cut. Note however that the number of samples per cluster might still have a considerable variability, since the splitting of a big cluster can result in sub-clusters with very variable number of samples. This variability should be smaller as the given maximum of samples per cluster is closer to 1 (being the variability equal to 0 when the maximum is at its limit, i.e. 1).

The function returns two results: 
1. List of integers containing for each input sample its corresponding cluster id. The cluster id is an integer which is higher for deeper tree levels.
2. List of strings containing for each input sample its corresponding cluster tree level, i.e. a string of '0's and '1's separated by '.' Note that the cluster level is longer for deeper tree levels, being 0 the root cluster, 0.0 and 0.1 its offspring, and so on.

# Installation

Simply install the scipy and numpy packages. It should run out of the box.

# Example Script

Please use the following command in order to run the example script.

```
$ python3 ward_cut_tree_balanced.py
```