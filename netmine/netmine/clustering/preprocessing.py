import numpy as np

class NanValuesHandler:
    """
    A base class for handling nan values in a dataset

    This class performs no handling; it returns the dataset as it is. It should be overriden to create nan
    value handler, in which case, at least handle and inverse_handle should be overriden.

    Attributes
    ----------
    nan_rows_indices : ndarray
        a 1d ndarray containing the indices of rows containing nan values. This can be used directly as an
        index of the dataset to set/get the nan containing rows.
    num_rows_indices : ndarray
        a 1d ndarray containing the indices of rows that does not contain nan values. This can be used directly
        as an index of the dataset to set/get the rows not containing nan.
    splitted_nan_indices : ndarray
        a 2d np array with each row containing 2 elements representing the location of one of the nan values in
        the fitted dataset.

    methods
    -------
    fit(dataset) : None
        finds the locations of nan values and sets indices and splitted_nan_indices attributes accordingly
    handle(dataset) :  ndarray
        handles a dataset nan values and returns the resulting dataset
    inverse_handle(dataset) : ndarray
        inversts the effect of handle
    fit_handle(dataset) : ndarray
        finds and handles a dataset nan values and returns the resulting dataset
    """

    def __init__(self):
        """
        initializes object attributes

        nan_rows_indices, num_rows_indices and splitted_nan_inidces are set to None.
        """
        self.nan_rows_indices = []
        self.num_rows_indices = None
        self.splitted_nan_indices = None

    def fit(self, dataset):
        """
        finds the locations of nan values

        It sets nan_rows_indices, num_indices and splitted_nan_indices attributes accordingly.

        :param ndarray dataset: the dataset searched for nan values.
        """
        nan_indices = np.isnan(dataset)
        self.nan_rows_indices = np.where(nan_indices.any(axis=1))[0]
        self.num_rows_indices = np.where(~(nan_indices.any(axis=1)))[0]
        self.splitted_nan_indices = np.array(np.where(nan_indices)).T

    def handle(self, dataset):
        """
        handles a dataset nan values and returns the resulting dataset

        This method assumes that nan values are located in the locations specified by nan_rows_indices
        attribute and hence it should not be called before calling fit. This method should be implemented by
        any class extending this class. Its implementation should allow for inverse_handle the ability to
        recover the dataset.

        :param ndarray dataset: the dataset whose nan values are to be handled
        :return: a nan-handled version of the dataset
        """
        pass

    def inverse_handle(self, dataset, default=np.nan):
        """
        recovers a dataset whose nan values has been handles

        This method assumes that nan values are located in the locations specified by nan_rows_indices attribute and
        and hence it should not be called before calling fit. This method should be implemented by any class
        extending this class. It allows the dataset to have added columns (at the end of the dataset) to
        the dataset passed to handle. The values at these columns are set to default.

        :param ndarray dataset: the dataset whose nan values has been handled
        :param int or ndarray default: the default value/values to be used to fill additional columns. if int,
        the value will be used for all entries. If 1d ndarray, each value in the array will be used for a
        column respectively; so, it should have a dimension equals to the number of additional columns. If 2d
        array, the array will be concatenated to the dataset; i.e, each value in the array will be mapped to
        the corresponing location in the extension matrix added to the dataset; so, its dimension should be
        (m,n) such that m is the number of rows containing nan values while n is the number of additional
        columns.
        :return: the original dataset before calling handle
        """
        pass

    def fit_handle(self, dataset):
        """
        handles a dataset nan values and returns the resulting dataset

        This method does not assume that nan values are located in the locations specified by nan_rows_indices
        attribute like handle. It is equivalent to calling fit and then handle.

        :param ndarray dataset: the dataset whose nan values are to be handled
        :return: a nan-handled version of the dataset
        """
        self.fit(dataset)
        return self.handle(dataset)

class IgnoreNanValuesHandler(NanValuesHandler):
    """
    Nan values handler that ignores all rows with nan values

    Attributes
    ----------
    nan_rows : ndarray
        a 2-d numpy array that holds the values of the rows containing nan values from the dataset passed to
        handle. This is used to allow recovering the dataset with inverse_handle.

    Methods
    -------
    handle : ndarray
        handles the nan values by removing the rows containing nan values.
    inverse_handle : ndarray
        recovers the dataset handled by handle by inserting back the removed rows.
    """
    def __init__(self):
        """
        initializes the attributes of the object

        nan_rows is set to None.
        """
        super().__init__()
        self.nan_rows = None

    def handle(self, dataset):
        """
        handles the nan values by removing the rows containing nan values.

        This method overrides the base class methood. It set nan_rows attribute accordingly to allow recovery
        via inverse_handle. It uses the nan_rows_indices and num_rows_indices set by fit; so, it should not be
        called before calling fit. Also, to give relevant results it should be called with a dataset with rows
        containing nan values in the same locations of those in the dataset passed to fit. Idealy, the dataset
        passed to fit should also passed to this method. It is more natural to call fit_handle rather than
        calling hit then handle.

        :param ndarray dataset: the dataset whose nan values are to be handled
        :return: a nan-handled version of the dataset
        """
        self.nan_rows = dataset[self.nan_rows_indices]
        return dataset[self.num_rows_indices]

    def inverse_handle(self, dataset, default=np.nan):
        """
        recovers a dataset whose nan values have been handles

        This method overrides the base class method. This method assumes that nan values are located in the
        locations specified by nan_rows_indices attribute and hence it should not be called before calling fit.
        It uses nan_rows attribute set by handle; hence, it should not be called before handle. It allows the
        dataset to have added columns (at the end of the dataset) to the dataset passed to handle. The values
        at these columns is set according to default parameter.

        :param ndarray dataset: the dataset whose nan values has been handled
        :param int or ndarray default: the default value/values to be used to fill additional columns. if int,
        the value will be used for all entries. If 1d ndarray, each value in the array will be used for a
        column respectively; so, it should have a dimension equals to the number of additional columns. If 2d
        array, the array will be concatenated to the dataset; i.e, each value in the array will be mapped to
        the corresponing location in the extension matrix added to the dataset; so, its dimension should be
        (m,n) such that m is the number of rows containing nan values while n is the number of additional
        columns.
        :return: the original dataset before calling handle
        """
        extended_nan_rows = np.zeros((self.nan_rows.shape[0],dataset.shape[1]))
        extension_of_nan_rows = np.zeros((self.nan_rows.shape[0],dataset.shape[1] - self.nan_rows.shape[1]))
        extended_nan_rows[:] = default
        extended_nan_rows = np.concatenate((self.nan_rows,extension_of_nan_rows), axis=1)

        dataset = np.insert(dataset, obj=self.nan_rows_indices - np.arange(self.nan_rows_indices.shape[0]),
                            values=extended_nan_rows, axis=0)
        return dataset

class Scaler:
    """
    Base class for scaler classes
    
    Methods
    -------
    fit : None
        prepares the scaling parameters according to the dataset
    handle : ndarray
        scales a dataset based on the parameters specified in fit
    fit_handle : ndarray
        prepares the scaling parameters and scale a datatset
    inverse_handle : ndarray
        inversts the effect of handle        
    """

    def fit(self, dataset):
        """
        prepares the scaling parameters according to the dataset
        
        This method should be overriden in any class that extends this class.

        :param ndarray dataset: the dataset to set the parameters according to 
        """
        pass

    def handle(self, dataset):
        """
        scales a dataset based on the parameters specified in fit
        
        This method should be overriden in any class that extends this class. This method should not be
        called before calling fit with a dataset with similar features. Similar features is specified 
        according to the specific scaling method.

        :param ndarray dataset: The dataset to be scaled.
        :return ndarray: The scaled version of the dataset. 
        """
        pass

    def fit_handle(self, dataset):
        """
        prepares the scaling parameters and scale a datatset
        
        Calling this method is equivalent to calling fit and then handle with the same dataset.

        :param ndarray dataset: The dataset to be scaled.
        :return ndarray: The scaled version of the dataset.
        """
        self.fit(dataset)
        return self.handle(dataset)

    def inverse_handle(self, dataset):
        """
        inversts the effect of handle
        
        This method should be overriden in any class that extends this class. calling this method on 
        the result of handle should returns the original dataset passed to handle.

        :param ndarray dataset: the scaled dataset whose scaling to be inverted
        :return ndarray: the recoovered dataset 
        """
        pass

class SigmoidWith10pecrentScaler(Scaler):
    """
    A scaler class that uses sigmoid function
    
    This class multiplies the dataset by a sigmoid function such that 2 specific points are mapped to
    approximately 10% and 90% - of the entire resulting range - after scaling.
    
    Methods
    -------
    
            
    """

    def __init__(self, a10=10, a90=90):
        """
        :param a10: The value/values to be mapped to 10% point in the resulting range. If the dataset
        to be scaled contains many columns, then, each coulumn may have a different value for a10 and
        a90; therefore, a10 and a190 are stored as ndarrays with a length equals to the number of the
        columns. Or, if the values are the same regardless of the column used, a single values for
        a10 and a90 can be used.
        :param a90: The value/values to be mapped to 10% point in the resulting range. An argument
        similar to that written for a10 applies here.
        """
        self.a10 = a10
        self.a90 = a90
        self._alpha = 0
        self._beta = 1

    def fit(self, dataset):
        """
        prepares _alpha and _beta parameters according to a10 and a90

        _alpha and _beta are used as parameters for the sigmoid function. They are set here such that
        a10 values give .1 and a90 values give .9 after scaling. This method shoud be calld whenever
        the values of a10 or a90 changes.

        :param dataset: this is passed for compatibility reasons
        :return:
        """
        self._alpha = (self.a10 + self.a90) / 2.
        self._beta = (self.a90 - self._alpha) / 2.
        
    def handle(self, dataset):
        """
        Scales the dataset using a sigmoid function

        This method scales a dataet using the parameters set in fit. It overrides the base class
        method. It should not be called before calling fit.

        :param ndarray dataset: The dataset to be scaled
        :return ndarray: The scaled version of the dataset
        """
        dataset = dataset.copy()
        dataset = self._map_values(dataset)
        dataset = 1 / (1 + np.exp(-1 * dataset))
        return dataset

    def inverse_handle(self, dataset):
        """
        inversts the effect of handle using the inverse of the sigmoid function

        This method scales a dataet using the parameters set in fit. It overrides the base class
        method.

        :param ndarray dataset: the scaled dataset whose scaling to be inverted
        :return ndarray: the recoovered dataset
        """
        dataset = dataset.copy()
        dataset = np.log((1 / dataset) - 1) * -1
        dataset = self._invere_map_values(dataset)
        return dataset

    def _map_values(self, x):
        """
        Linearly maps vaues using _alpha and _beta parameters

        :param int or ndarray x: The value/values to be mapped.
        :return: A mapped version of the dataset.
        """
        return (x - self._alpha) / self._beta

    def _invere_map_values(self, x):
        """
        Inversts the linear mapping made by_map_values

        _alpha and _beta should not change between calling_map_values and calling this method.

        :param int or ndarray x: The value/values to be mapped.
        :return: A mapped version of the dataset.
        """
        return x * self._beta + self._alpha
