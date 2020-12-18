import unittest
import numpy as np
from netmine.clustering.optimization import Optimizer

class OptimizerTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.choices = [9, 4, 5, 1, 3, 7, 6]
        self.filter = lambda x: x % 2 != 0
        self.criteria = lambda x: np.abs(x - 6)
        self.optimizer = Optimizer(self.choices, self.filter, self.criteria)

    def test_optimize(self):
        self.optimizer.optimize()
        self.assertListEqual(self.optimizer.accepted_choices.tolist(), [5,7,9,3,1])

    def test_adding_element(self):
        self.optimizer.optimize()
        self.optimizer.add_choice(11)
        self.optimizer.optimize()
        self.assertListEqual(self.optimizer.accepted_choices.tolist(), [5, 7, 9, 3, 11, 1])

    def test_adding_elements(self):
        self.optimizer.optimize()
        self.optimizer.add_choices([11,15])
        self.assertTrue(self.optimizer._updated)
        self.assertListEqual(self.optimizer._unprocessed_choices, [11,15])
        self.optimizer.optimize()
        self.assertListEqual(self.optimizer._unprocessed_choices, [])
        self.assertListEqual(self.optimizer.accepted_choices.tolist(), [5, 7, 9, 3, 11, 1, 15])

    def test_opimization_after_change(self):
        self.optimizer.optimize()
        self.optimizer.set_filter(lambda x: x > 4)
        self.assertFalse(self.optimizer._updated)
        self.optimizer.optimize()
        self.assertTrue(self.optimizer._updated)
        self.assertListEqual(self.optimizer.accepted_choices.tolist(), [6, 5, 7, 9])

if __name__ == '__main__':
    unittest.main()
