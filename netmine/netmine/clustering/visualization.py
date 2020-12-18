import numpy as np
import matplotlib.pyplot as plt



def plot_dataset(dataset, names, title='dataset'):
    """
    Plots a dataset 2 features at a time

    The first feature is plotted as the x axis in all of the plots, while the y axis changes from one
    plot to the other to cover all the plots.

    :param ndarray dataset: The dataset to be plotted
    :param list names: The names used for the axes
    :param str title: The title used for the plots
    """
    num_of_features = dataset.shape[1]
    for i in range(1, num_of_features):
        plt.figure()
        plt.title(title)
        plt.xlabel(names[0])
        plt.ylabel(names[i])
        plt.scatter(dataset[:,0], dataset[:,i])

def plot_clustering(dataset, labels, names, title='clustering'):
    """
    Plots a dataset 2 features at a time with each cluster with a different color

    The first feature is plotted as the x axis in all of the plots, while the y axis changes from one
    plot to the other to cover all the plots. Different colors are usd for different clusters.

    :param ndarray dataset: The dataset to be plotted
    :param ndarray labels: The labels based on which the data points are colored
    :param list names: The names used for the axes
    :param str title: The title used for the plots
    """
    num_of_features = dataset.shape[1]
    for i in range(1, num_of_features):
        plt.figure()
        plt.title(title)
        plt.xlabel(names[0])
        plt.ylabel(names[i])
        for label in np.unique(labels):
            plt.scatter(dataset[labels==label,0], dataset[labels==label,i])

def plot_som_map(labels, som, title="som map"):
    """
    Plots a map for SOM contining the data points.

    :param ndarray labels: The labels that specify where each data point is to be plotted
    :param minisom.MiniSom som: The SOM whose map is to be plotted
    :param str title: The title used for the plot
    """
    plot_som_map_mapping(labels, labels, som, title)

def plot_som_map_mapping(labels, som_labels, som, title='som mapping'):
    """
    Plots a map for SOM contining the data points colored based on som-based clustering.

    This shows which SOM neurons where put into the same cluster by looking to the colors of the points
    inside each neuron space in the map.

    :param ndarray labels: The labels that specify the color of each data point
    :param ndarray som_labels: The labels that specify where each data point is to be plotted
    :param minisom.MiniSom som: The SOM whose map is to be plotted
    :param str title: The title used for the plot
    """
    dist_map = som.distance_map().T
    hight, width = dist_map.shape

    som_labels_x = som_labels // hight
    som_labels_y = som_labels % hight

    plt.figure()
    plt.pcolor(dist_map, cmap='bone_r', alpha=.2)
    plt.title(title)
    plt.xlabel("som's map x axis")
    plt.ylabel("som's map y axis")
    plt.xlim((0, width))
    plt.ylim((0, hight))
    plt.xticks(range(width))
    plt.yticks(range(hight))

    for som_label in np.unique(som_labels):
        filter = som_labels == som_label
        label = labels[filter][0]
        plt.scatter(som_labels_x[filter] + .5 + (np.random.rand(np.sum(filter)) - .5) * .8,
                    som_labels_y[filter] + .5 + (np.random.rand(np.sum(filter)) - .5) * .8,
                    s=50, c='C'+str(label))
    plt.grid()

def plot_clusters_features_histograms(dataset, names, title=''):
    """
    plots histgrams for all of the features in a dataset

    :param dataset: The dataset whose data histograms are plotted
    :param names: The names used for the x axes of the histograms
    :param title: The title of the plots
    """
    num_of_features = dataset.shape[1]
    for i in range(num_of_features):
        plt.figure()
        plt.title(title)
        plt.xlabel(names[i])
        plt.hist(dataset[:, i], bins=100, rwidth=.6, density=True)