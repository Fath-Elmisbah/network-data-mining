import unittest
import numpy as np
from netmine.clustering.optimization import Optimizer
from netmine.clustering.state_io import store_dill, load_dill

class OptimizerTestCase(unittest.TestCase):

    def setUp(self) -> None:
        """
        Prepares an optimizer object with choices, a filter that filters even numbers out, and a criteria
        that measures the distance from 6
        """
        self.choices = [9, 4, 5, 1, 3, 7, 6]
        self.filter = lambda x,y: x % 2 != 0
        self.criteria = lambda x,y: np.abs(x - 6)
        self.optimizer = Optimizer(self.choices, self.filter, self.criteria)

    def test_optimize(self):
        """
        tests optimize function against the normal input
        """
        self.optimizer.optimize()
        self.assertListEqual(self.optimizer.accepted_choices.tolist(), [7,5,3,9,1])

    def test_adding_element(self):
        """
        tests whether added elements with add_choice are inserted correctly in the sorted accepted_choices
        """
        self.optimizer.optimize()
        self.optimizer.add_choice(11)
        self.optimizer.optimize()
        self.assertListEqual(self.optimizer.accepted_choices.tolist(), [7., 5, 3, 9., 11., 1.])

    def test_adding_elements(self):
        """
        tests whether added elements with add_choices are inserted correctly in the sorted accepted_choices
        """
        self.optimizer.optimize()
        self.optimizer.add_choices([11,15])
        self.assertTrue(self.optimizer._updated)
        self.assertListEqual(self.optimizer._unprocessed_choices, [11,15])
        self.optimizer.optimize()
        self.assertListEqual(self.optimizer._unprocessed_choices, [])
        self.assertListEqual(self.optimizer.accepted_choices.tolist(), [7, 5, 3, 9, 11, 1, 15])

    def test_opimization_after_change(self):
        """
        tests whether changning filter forces optimize to work from the begining
        """
        self.optimizer.optimize()
        self.optimizer.set_filter(lambda x,y: x > 4)
        self.assertFalse(self.optimizer._updated)
        self.optimizer.optimize()
        self.assertTrue(self.optimizer._updated)
        self.assertListEqual(self.optimizer.accepted_choices.tolist(), [6, 7, 5, 9])

    def test_optimization_store_load(self):
        self.optimizer._store = store_dill
        path = 'test_optimization\\store_state.pkl'
        self.optimizer.optimize(store_filename=path)
        new_optimizer = Optimizer.from_io(Optimizer(),load_dill, path)
        self.assertListEqual(new_optimizer.accepted_choices.tolist(), [7, 5, 3, 9, 1])

if __name__ == '__main__':
    unittest.main()
