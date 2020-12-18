import numpy as np
from minisom import MiniSom
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from netmine.clustering import utils


class Clustering:
    """
    A base class for all clustering classes

    ...
    Attributes
    ----------
    name : str
        a human readable name for the type of clustering
    num_of_clusters : int
      the number of clusters produced

    Methods
    -------
    fit(dataset)
      trains the clustering object using a dataset and sets the dimensions of the data to be classified
    predict(dataset)
      classifies a dataset based on the training made through fit
    fit_predict(dataset)
      trains the clustering object using a dataset and classifies it using the object
    ...
    This class is build to be extended. For any class extending this class, it should either override name
    attribute or __str__ method. Also, it should implement fit and predict methods, and optionally, the class
    can implement fit_predict method to provide more efficient implementation
    """

    name = 'base clustering'

    def __init__(self):
        """
        Initializes the object by setting num_of_clusters to 0
        """
        self.num_of_clusters = 0

    def __str__(self):
        return 'name : ' + self.name + '\t' + 'number of clusters : ' + self.num_of_clusters

    def fit(self, dataset):
        """
        Trains the clustering object using a dataset, and sets the dimensions of the data to be classified

        This method should be overriden by any class extending this class.

        :param ndarray dataset: the dataset used to train the clustering object. This should be numpy 2d array.
        """
        pass

    def predict(self, dataset):
        """
        Clusters a dataset based on the training made through fit

        This method should be overriden by any class extending this class, and it should not be called before
        calling fit function with a dataset with the same number of features of the dataset passed here.

        :param ndarray dataset: the dataset to be custeredby  the clustering object. This should be numpy 2d
        array.
        """
        pass

    def fit_predict(self, dataset):
        """
        Trains the clustering object using a dataset and classifies it using the object

        Calling this method is equivalent to calling fit and then predict with the same dataset.

        :param ndarray dataset: the dataset to be clustered by the clustering object. This should be numpy 2d
        array.
        :returns: ndarray labels: an array containing the cluster labels.
        """
        self.fit(dataset)
        return self.predict(dataset)


class SOMClustering(Clustering):
    """
    A class that performs clustering based on Self-Organizing-Map neural networks

    It perorms the clustering through 2 main phases: a rough phase, and a fine phase. rough phase is used to
    perfotm a fast training that puts each wieght vector near to its optimal location.
    Base class: netmine.clustering.clustering.Clustering.

    Attributes
    ----------
    som : minisom.MiniSom
        the object that represents the SOM neural network
    centroids : ndarray
        an array of the weight vectors of the som network neurons

    Methods
    -------
    fit(dataset)
      trains the SOM neural network using a dataset
    predict(dataset)
      clusters a dataset based on the SOM network trained through fit method
    """
    name = 'SOM clustering'

    def __init__(self, width=11, hight=11, sigma1=None, alpha1=1, num_of_iters1=1000, sigma2=1, alpha2=.01,
                 num_of_iters2=20000, initialize=None):
        """
        Initialize the object attributes

        num_of_clusters is set to width*hight where som and centroids are both set to None.

        :param int width: the width of the SOM network. (default 11)
        :param int hight: the hight of the SOM network. (default 11)
        :param float sigma1: the neighborhood length of the SOM network in the rough phase. If None, It is set
        to the minimum of width -2 and hight - 2 if the minimum is positive and to 1 otherwise. (default None)
        :param float alpha1: the learning rate of the som network in the rough phase. (default 1)
        :param int num_of_iters1: the number of iterations of the rough phase. (default 1000)
        :param float sigma2: the neighborhood length of the SOM network in the fine phase. (default 1)
        :param float alpha2: the learning rate of the som network in the fine phase. (default .01)
        :param int num_of_iters2: the number of iterations of the fine phase. (default 20000)
        :param function initialize: a function used to initialize the weights of the som. If None, PCA
        initialization will be used. (default None)
        """
        self.width = width
        self.hight = hight
        self.num_of_clusters = self.width * self.hight
        if sigma1 == None:
            self.sigma1 = max(1,min(self.width, self.hight) - 2)
        else:
            self.sigma1 = sigma1
        self.alpha1 = alpha1
        self.num_of_iters1 = num_of_iters1
        self.sigma2 = sigma2
        self.alpha2 = alpha2
        self.num_of_iters2 = num_of_iters2
        if initialize == None:
            self.initialize = MiniSom.pca_weights_init
        else:
            self.initialize = initialize
        self.som = None
        self.centroids = None

    def fit(self, dataset):
        """
        Trains the SOM neural network using a dataset

        This method overrides the base class method.

        :param ndarray dataset: The dataset used to train the SOM network.
        """
        self.som = MiniSom(self.width, self.hight, dataset.shape[1],
                           sigma=self.sigma1, learning_rate=self.alpha1, neighborhood_function="bubble")
        self.initialize(self.som, dataset)
        self.som.train_batch(dataset, num_iteration=self.num_of_iters1, verbose=True)
        self.som._learning_rate = self.alpha2
        self.som._sigma = self.sigma2
        self.som.neighborhood = self.som._gaussian
        self.som.train_batch(dataset, num_iteration=self.num_of_iters2, verbose=True)
        self.centroids = self.som.get_weights().reshape((-1, dataset.shape[-1]))

    def predict(self, dataset):
        """
        Clusters a dataset using the SOM neural network trained by fit

        This method overrides the base class method, and it should not be called before calling fit function
        with a dataset with the same number of features of the dataset passed here.

        :param ndarray dataset: the dataset to be clustered by the SOM network.
        :return ndarray labels: an array containing the cluster labels.
        """
        labels = []
        for _x in dataset:
            som_winner_neuron = self.som.winner(_x)
            label = som_winner_neuron[0] * self.hight + som_winner_neuron[1]
            labels.append(label)
        return np.array(labels)


class WardClustering(Clustering):
    """
    A class that performs clustering based on Wards Hierarchical method

    Attributes
    ----------
    ward_clustering : sklearn.clustering.AgglomerativeClustering
        an object performs Ward clustering
    dataset : ndarray
        an ndarray used to keep the dataset used to fit the clustering object

    Methods
    -------
    fit(dataset)
        trains the Ward clustering object using a dataset
    predict(dataset)
        trains the ward clustering object and clusters the dataset
    fit_predict(dataset)
        trains the ward clustering object and clusters the dataset
    """
    name = 'Ward Clustering'

    def __init__(self, num_of_clusters = 20):
        """
        Initializes the object attributes

        both ward_clustering and dataset are set to None.

        :param int num_of_clusters: the number of clusters to be produced
        """
        self.num_of_clusters = num_of_clusters
        self.ward_clustering = None
        self.dataset = None

    def fit(self, dataset):
        """
        Trains the Ward clustering object using a dataset

        This method overrides the base class method. It stores the passed dataset in the attribute dataset.

        :param ndarray dataset: the dataset used to train the Ward clustering object.
        """
        self.ward_clustering = AgglomerativeClustering(n_clusters=self.num_of_clusters, affinity='euclidean', linkage='ward')
        self.ward_clustering.fit(dataset)
        self.dataset = utils.encode_2d_in_1d_array(dataset)

    def predict(self, dataset):
        """
        Trains the Ward clustering object using a dataset and clusters it

        This method overrides the base class method. It uses the ward clustering object trained by fit. This
        method should not be called before calling fit with a dataset with the same number of features of the
        dataset used in fit. Additionally, if a datapoint does not present in the dataset used in fit, the
        resulting label will be replaced by -1.

        :param ndarray dataset: the dataset to be clustered by the Ward clustering object.
        :return ndarray labels: an array containing the cluster labels or -1 if the data point does not
        present in the dataset used in fit.
        """
        dataset = utils.encode_2d_in_1d_array(dataset).reshape((-1,1))
        data_indices, original_indices = np.where(dataset==self.dataset)
        mapped_indices = np.zeros(dataset.shape[0]) - 1
        mapped_indices[data_indices] = original_indices
        labels = self.ward_clustering.labels_[mapped_indices.astype(int)]
        labels[np.where(mapped_indices==-1)] = -1
        return labels

    def fit_predict(self, dataset):
        """
        Trains the Ward clustering object using a dataset and clusters it

        This method overrides the base class method.

        :param ndarray dataset: the dataset to be clustered by the Ward clustering object.
        :return ndarray labels: an array containing the cluster labels.
        """
        self.ward_clustering = AgglomerativeClustering(n_clusters=self.num_of_clusters, affinity='euclidean',
                                                       linkage='ward')
        return self.ward_clustering.fit_predict(dataset)


class KmeansClustering(Clustering):
    """
    A class that performs k-means clustering

    Attributes
    ----------
    k_means_clustering : sklearn.clustering.KMeans
        an object performs k-means clustering

    Methods
    -------
    fit(dataset)
        trains the k-means clustering object using a dataset
    predict(dataset)
        clusters a dataset based on the k-means clustering object trained through fit method
    fit_predict(dataset)
        trains the k-means clustering object and clusters the dataset
    """

    def __init__(self, num_of_clusters, num_of_iters=1000, num_of_trials=10):
        """
        initializes the object attributes

        k_means_clustering is set to None

        :param int num_of_clusters: the number of clusters required
        :param int num_of_iters: the number of iterations per trial
        :param int num_of_trials: the number of trials of run from which the best is selected
        """
        self.num_of_clusters = num_of_clusters
        self.num_of_iters = num_of_iters
        self.num_of_trial = num_of_iters
        self.k_means_clustering = None

    def fit(self, dataset):
        """
        Trains the k-means clustering object using a dataset

        This method overrides the base class method

        :param ndarray dataset: The dataset used to train the k-means object.
        """
        self.k_means_clustering = KMeans(n_clusters=self.num_of_clusters, n_init=self.num_of_trial,
                                         max_iter=self.num_of_iters)
        self.k_means_clustering.fit(dataset)

    def predict(self, dataset):
        """
        Clusters a dataset using the k-means clustering object trained by fit

        This method overrides the base class method, and it should not be called before calling fit function
        with a dataset with the same number of features of the dataset passed here.

        :param ndarray dataset: the dataset to be clustered by the k-means clustering object.
        :return ndarray labels: an array containing the cluster labels.
        """
        return self.k_means_clustering.predict(dataset)

    def fit_predict(self, dataset):
        """
        Trains the k-means clustering object using a dataset and clusters it

        This method overrides the base class method.

        :param ndarray dataset: the dataset to be clustered by the k-means clustering object.
        :return ndarray labels: an array containing the cluster labels.
        """
        self.k_means_clustering = KMeans(n_clusters=self.num_of_clusters, n_init=self.num_of_trial,
                                         max_iter=self.num_of_iters)
        return self.k_means_clustering.fit_predict(dataset)


class SOMBasedClusetring(Clustering):
    """
    A class that performs 2-layers clustering with SOM clustering layer as the first stage

    Attributes
    ----------
    som_labels : ndarray
        som labels of the last dataset predicted.

    Methods
    -------
    fit(dataset)
        trains the the 2 layers of clustering using a dataset
    predict(dataset)
        clusters a dataset based on the 2 layers of clustering trained through fit method
    fit_predict(dataset)
        trains thethe 2 layers of clustering and clusters the dataset
    """

    def __init__(self, som_clustering, post_som_clustering):
        """
        :param SOMClustering som_clustering: the som clustering object used in the first layer
        :param Clustering post_som_clustering: the clustering object used in the second layer
        """
        self.som_clustering = som_clustering
        self.post_som_clustering = post_som_clustering
        self.som_labels = None

    def fit(self, dataset):
        """
        Trains the 2 layers of clustering using a dataset

        This method overrides the base class method. The 2nd layer clustering object is trained via the
        centroids of the trained som.

        :param ndarray dataset: The dataset used to train the 2 layers of clustering.
        """
        self.som_clustering.fit(dataset)
        self.post_som_clustering.fit(self.som_clustering.centroids)

    def predict(self, dataset):
        """
        Clusters a dataset using the 2 layers of clustering trained by fit

        This method overrides the base class method, and it should not be called before calling fit function
        with a dataset with the same number of features of the dataset passed here. It sets the attribute
        som_labels according to the first part of the clustering

        :param ndarray dataset: the dataset to be clustered by the 2 layers of clustering.
        :return ndarray labels: an array containing the cluster labels.
        """
        self.som_labels = self.som_clustering.predict(dataset)
        som_labels_weights = self.som_clustering.centroids[self.som_labels]
        labels = self.post_som_clustering.predict(som_labels_weights)
        return labels

    def fit_predict(self, dataset):
        """
        Trains the 2 layers of clustering using a dataset and clusters it

        This method overrides the base class method.  It sets the attribute som_labels
        according to the first part of the clustering

        :param ndarray dataset: the dataset to be clustered by the 2 layers of clustering.
        :return ndarray labels: an array containing the cluster labels.
        """
        self.som_labels = self.som_clustering.fit_predict(dataset)
        self.post_som_clustering.fit(self.som_clustering.centroids)
        som_labels_weights = self.som_clustering.centroids[self.som_labels]
        labels = self.post_som_clustering.predict(som_labels_weights)
        return labels

