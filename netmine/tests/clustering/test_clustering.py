import unittest
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from netmine.clustering.clustering import Clustering, SOMClustering, WardClustering, SOMBasedClusetring
from minisom import MiniSom

import utils

class SOMClusteringTestCase(unittest.TestCase):

    def test_sigm1_default_initalization(self):
        """
        Checks whether the argument sigma1 is initialized correctly when the default value is used

        It uses different initializations for the width and hight parameters. One with width > hight > 3, one
        with hight > width > 3 and one with hight > width with width < 3.

        Sigma1 should be the minimum of the two minus 2 given that sigma1 is not less than 1.
        """
        som_clustering = SOMClustering(width=10, hight=8)
        self.assertEqual(som_clustering.sigma1, 6, msg='sigma1 should equal hight - 2 = 6 because hight is '
                                                            'less than width')
        som_clustering = SOMClustering(width=7, hight=8)
        self.assertEqual(som_clustering.sigma1, 5, msg='sigma1 should equal width - 2 = 5 because width is '
                                                            'less than hight')
        som_clustering = SOMClustering(width=2, hight=8)
        self.assertEqual(som_clustering.sigma1, 1, msg='sigma1 should equal 1 because width - 2 = 0 is less'
                                                            ' than 1')

    def test_initialize_arg_initialized_with_None(self):
        """
        Checks whether the argument initialize is initialized correctly when the None is used

        Initialize should be the initialized to pca_weights_init method in MiniSom class
        """
        som_clustering = SOMClustering(initialize=None)
        self.assertEqual(som_clustering.initialize, MiniSom.pca_weights_init)

    def test_fit_check_parameters_for_fine_phase(self):
        """
        Checks whether the som network is trained in the fine phase with the corresponding parameters

        Som parameters after fit should be alpha2 for _learning_rate, sigma2 for _sigma, and _gaussian function
        of MiniSom for neighborhood.
        """
        som_clustering = SOMClustering(hight=5, width=5, sigma2=1, sigma1=3, alpha1=1, alpha2=.01,
                                       num_of_iters1=100,num_of_iters2=200)
        data = np.random.random((3,2))
        som_clustering.fit(dataset=data)
        self.assertEqual(som_clustering.som._sigma, 1, msg='sigma should change to sigma2 value during the '
                                                           'training')
        self.assertEqual(som_clustering.som._learning_rate, .01, msg='learning rate should change to alpha2 '
                                                                   'value during the training')
        self.assertEqual(som_clustering.som.neighborhood, som_clustering.som._gaussian)

    def test_predict_check_labels(self):
        """
        Checks whether the labels produced by predict are consistant with MiniSom conventions

        It uses a weight from the SOM network itself and finds the nearest weight to it according to predict
        method and MiniSom get_weights method.

        The weight produced should be the same weight fetched to the network.
        """
        som_clustering = SOMClustering(hight=5, width=5, sigma2=1, sigma1=3, alpha1=1, alpha2=.01,
                                       num_of_iters1=100, num_of_iters2=200)
        data = np.random.random((3,2))
        som_clustering.fit(dataset=data)
        weights = som_clustering.centroids
        weight = weights[0:1]
        self.assertListEqual(list(weights[som_clustering.predict(weight)[0]]),list(weight[0]))

    def test_fit_predict_simple_clustering_task(self):
        """
        Checks whether SOM performs basic clustering for only 4 data points with 4 neurons

        The data points should be classified into 4 different classes. Moreover, the data points near to
        eachother should by classified by neighboring neurons, and those far away from eachother should be
        classified by opposing neurons.
        """
        som_clustering = SOMClustering(hight=2, width=2, sigma2=1, sigma1=2, alpha1=.1, alpha2=.01,
                                       num_of_iters1=100, num_of_iters2=200)
        data = np.array([[10, 10], [-10, -10], [10, -10], [-10, 10]])
        labels = som_clustering.fit_predict(data)
        bin_labels = list(map(utils.convert_to_binary,labels,[2]*4))
        print(bin_labels)
        # ensure that [1, 1] is classified with the furthest neuron from that of [-1,-1]
        self.assertNotEqual(bin_labels[0][0], bin_labels[1][0])
        self.assertNotEqual(bin_labels[0][1], bin_labels[1][1])
        # ensure that [1, -1] is classified with the furthest neuron from that of [-1,1]
        self.assertNotEqual(bin_labels[2][0], bin_labels[3][0])
        self.assertNotEqual(bin_labels[2][1], bin_labels[3][1])
        # ensure that [1, 1] is classified with the nearest neuron from that of [-1,1]
        self.assertTrue(bin_labels[0][0] == bin_labels[3][0] or bin_labels[0][1] == bin_labels[3][1])
        self.assertFalse(bin_labels[0][0] == bin_labels[3][0] and bin_labels[0][1] == bin_labels[3][1])

class WardClusteringTestCase(unittest.TestCase):

    def test_predict_normal_input(self):
        """
        Checks Ward clustering predict against a normal input

        When Ward clustering fit and predict methods are called seperately for the same datased, thier overall
        impact should be the same as if fit_predict is called for the dataset. This test is made because
        fit_predict in the case of Ward clustering is not just calls for fit and the predict methods. In fact,
        predict is not provided by the API of AgglomerativeClustring; hence, it is implemented here. A normal
        input is used here; the same dataset used for fit is used for predict.
        Predict after fit should give the same result of fit_predict.
        """
        ward_clustering = WardClustering(3)
        data = np.array([[0,0],[0,1],[120,2],[150,3],[-10,-19],[-30,-20]])
        ward_clustering.fit(data)
        found_labels = ward_clustering.predict(data)
        expected_labels = AgglomerativeClustering(3).fit_predict(data)
        self.assertListEqual(list(found_labels),list(expected_labels))

    def test_abnormal_strange_input(self):
        """
        Checks Ward clustering predict against an abnormal input

        Two datasets are used: one for fit and the other for predict. the dataset used for predict contains
        data points that does not present in the first dataset. For these data points, predict should return
        -1. For the other ones, predict should return the correct label.
        """
        ward_clustering = WardClustering(3)
        data = np.array([[0, 0], [0, 1], [120, 2], [150, 3], [-10, -19], [-30, -20]])
        data2 = np.array([[0,1],[0,1],[0,0],[3,3]])
        ward_clustering.fit(data)
        found_labels = ward_clustering.predict(data2)
        expected_labels = AgglomerativeClustering(3).fit_predict(data)
        self.assertListEqual(list(found_labels), [expected_labels[1],expected_labels[1],expected_labels[0],-1])

class SOMBasedClusteringTestCase(unittest.TestCase):

    def test_fit_predict(self):
        """
        Checks whether fit_predict works correctly by using trained som centroids

        Som centroids should be classified by the overall clustering the same as its classification with the 2nd
        layer only. because som produces the smae centroid when it is applied to it.
        """
        data = np.random.random((15,2))
        som_clustering = SOMClustering(2,2,num_of_iters1=100,num_of_iters2=200)
        ward_clustering = WardClustering(3)
        som_ward_clustering = SOMBasedClusetring(som_clustering=som_clustering,post_som_clustering=ward_clustering)
        som_ward_clustering.fit(data)
        self.assertEqual(ward_clustering.predict(som_clustering.centroids[0:1]),
                         som_ward_clustering.predict(som_clustering.centroids[0:1]))

if __name__ == '__main__':
    unittest.main()
