import numpy as np

def order_clusters(labels):
    """
    Orders the clusters based on commonality

    This function takes labels vector/vectors. For each vector, every appearance of the most common label
    is converted to 0, the next is set to 1, and so on. In effect, The most common value in each vector
    would be 0 after calling this function.

    :param ndarray labels: The 1d or 2d ndarray whose labels are to be ordered.
    :return ndarray: 1d or 2d ndarray which repreents the ordered version of the labels array.
    """
    is1d = False
    if len(labels.shape) == 1:
        is1d = True
        labels.reshape((-1,1))
    nums_of_labels = np.max(labels, axis=0)
    results = np.zeros(labels.shape)
    m = np.max(nums_of_labels)
    n = labels.shape[1]
    counts = np.zeros((m+1,n))
    for i in range(m+1):
        counts[i,:] = np.sum(labels == i, axis=0)
    indices = np.flip(np.argsort(counts, axis=0), axis=0)
    for i in range(m+1):
        for j in range(n):
            results[np.where(labels[:,j] == indices[i,j]), j] = i
    if is1d:
        results.flatten()
        labels.flatten()
    return results