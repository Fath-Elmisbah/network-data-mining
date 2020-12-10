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
    splitted_indices : ndarray
        a 2d np array with each row containing 2 elements representing the location of one of the nan values in
        the fitted dataset.

    methods
    -------
    fit(dataset) : None
        finds the locations of nan values and sets indices and splitted_indices attributes accordingly
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

        nan_rows_indices, num_rows_indices and splitted_inidces are set to None.
        """
        self.nan_rows_indices = []
        self.num_rows_indices = None
        self.splitted_indices = None

    def fit(self, dataset):
        """
        finds the locations of nan values

        It sets nan_rows_indices, num_indices and splitted_indices attributes accordingly.

        :param ndarray dataset: the dataset searched for nan values.
        """
        nan_indices = np.isnan(dataset)
        self.nan_rows_indices = np.where(nan_indices.any(axis=1))[0]
        self.num_rows_indices = np.where(~(nan_indices.any(axis=1)))[0]
        self.splitted_indices = np.array(np.where(nan_indices)).T

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
