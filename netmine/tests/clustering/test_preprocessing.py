import unittest
import numpy as np
from netmine.clustering.preprocessing import NanValuesHandler, IgnoreNanValuesHandler

class NanValueHandlerTestCase(unittest.TestCase):

    def test_check_fit_normal_dataset(self):
        """
        Checks whether the features of NanValueHandlercare set correctly by fit

        It uses a dataset with nan values in 3 positions: (1,0), (2,1) and (6,2). The dataset contains 10 rows.

        nan_rows_indices should be [1,2,6], num_rows_indices should be its complement [0,3,4,5,7,8,9] and
        splitted_indices should be [[1,0],[2,1],[6,2]]
        """
        nan_value_handler = NanValuesHandler()
        data = np.random.random((10,4))
        rows = [1,2,6]
        rows_complement = [0,3,4,5,7,8,9]
        columns = [0,1,2]
        data[rows,columns] = np.nan
        nan_value_handler.fit(data)
        self.assertListEqual(nan_value_handler.nan_rows_indices.tolist(),rows)
        self.assertListEqual(nan_value_handler.num_rows_indices.tolist(),rows_complement)
        self.assertListEqual(nan_value_handler.splitted_indices.T.tolist(),[rows,columns])

    def test_check_fit_dataset_with_2_nans_in_a_single_row(self):
        """
        Checks whether the features of NanValueHandlercare are set correctly by fit

        It uses a dataset with nan values in 3 positions: (1,0), (1,1) and (6,2). Two of these values are in the same
        row.

        nan_rows_indices should be [1,6] not [1,1,6]
        """
        nan_value_handler = NanValuesHandler()
        data = np.random.random((8,3))
        rows = [1,1,6]
        expected = [1,6]
        rows_complement = [0,2,3,4,5,7]
        columns = [0,1,1]
        data[rows,columns] = np.nan
        nan_value_handler.fit(data)
        self.assertListEqual(nan_value_handler.nan_rows_indices.tolist(),expected)

    def test_check_fit_handle_dataset_without_nan(self):
        """
        Checks whether the features of NanValueHandlercare set correctly by fit

        The dataset used here does not contain any nan values.

        nan_rows_indices should be empty array.
        """
        nan_value_handler = NanValuesHandler()
        data = np.random.random((6,7))
        rows_complement = [0,1,2,3,4,5]
        nan_value_handler.fit(data)
        self.assertListEqual(nan_value_handler.nan_rows_indices.tolist(),[])

class IgnoreNanValuesHandlerTestCase(unittest.TestCase):

    def test_fit_handle_normal_dataset(self):
        """
        Checks whether handle removes the rows of nan from the dataset
        """
        ignore_nan_value_handler = IgnoreNanValuesHandler()
        data = np.random.random((10, 4))
        rows = [1, 2, 6]
        rows_complement = [0, 3, 4, 5, 7, 8, 9]
        columns = [0, 1, 2]
        data[rows, columns] = np.nan
        found_data = ignore_nan_value_handler.fit_handle(data)
        self.assertListEqual(found_data.tolist(),data[rows_complement].tolist())

    def test_handle_inverse_handle(self):
        """
        Checks whether handle and inverse_handle are complements
        """
        ignore_nan_value_handler = IgnoreNanValuesHandler()
        data = np.random.random((10, 4))
        rows = [1, 2, 6]
        rows_complement = [0, 3, 4, 5, 7, 8, 9]
        columns = [0, 1, 2]
        data[rows, columns] = np.nan
        found_data = ignore_nan_value_handler.fit_handle(data)
        found_data = ignore_nan_value_handler.inverse_handle(found_data)
        found_data[np.where(np.isnan(found_data))] = -100
        data[np.where(np.isnan(data))] = -100
        self.assertListEqual(found_data.tolist(), data.tolist())

if __name__ == '__main__':
    unittest.main()
