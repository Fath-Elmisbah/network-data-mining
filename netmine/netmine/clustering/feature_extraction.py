import numpy as np

class FeatureExtractor:
    """
    A base class for feature extraction classes

    This class performs no extraction; it returns the dataset as it is. It should be overriden to create feature
    extraction classes, in which case, at least fit method should be overriden.

    Attributes
    ----------
    features : ndarray
        array repreents the indices of the columns to be selected

    methods
    -------
    fit(dataset) : None
        selects the best features from a dataset and sets the attribute features accordingly
    extract(dataset) :  ndarray
        returns a dataset that represents a given one but with the extracted features
    fit_extract(dataset) : ndarray
        selects the best features from a dataset and return a modified version of the dataset
    """

    def __init__(self, num_of_features=None):
        """
        Initializes the object attributes

        features parameter is set to None.
        :param int num_of_features: the required number of features
        """
        self.num_of_features = num_of_features
        self.features = None

    def fit(self, dataset):
        """
        selects the best features from a dataset and sets the attribute features accordingly

        This method should be overriden to implement a feature extraction class. Currently it sets the features
        to be all the columns of the dataset.

        :param ndarray dataset: The dataset according which the features are selected.
        """
        self.features = np.arange(dataset.shape[1])

    def extract(self, dataset):
        """
        Extracts features from a dataset according to scheme decided through fit

        :param dataset: The dataset to apply feature extraction on.
        :return ndarray: A dataset that represents a given one but with the extracted features
        """
        return dataset[:, self.features]

    def fit_extract(self, dataset):
        """
        Selects the best features from a dataset and return a modified version of the dataset

        It extracts features from a dataset using a schema decided based on the dataset itself. It is equivalent
        to calling fit and then predict to the same dataset.

        :param dataset: The dataset to extract features from.
        :return ndarray: A dataset that represents a given one but with the extracted features
        """
        self.fit(dataset)
        return self.extract(dataset)