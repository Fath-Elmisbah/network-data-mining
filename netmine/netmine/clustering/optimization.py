import numpy as np

class Optimizer:
    """
    A class that performs general optimization

    Attributes
    ----------
    accepted_choices : list
        A list containing all objects that passed the filter test sorted by their scores descending
    rejected_choices : list
        A list containing all objects that could not pass the filter test
    scores : ndarray
        An 1d ndarray containing the scores of the accepted_choices in the same order
    """

    def __init__(self, choices=[], filter=lambda x,y: True, criteria=lambda x,y: 0):
        """
        :param list choices: The competing objects.
        :param function filter: A filter used to exclude some of the choices. It is a function that takes
        a choice with a generic parameter and gives a boolean value which specify whether to include the
        choice or not. If True, the choice is included.
        :param function criteria: The criteria used for optimization. It is a function that takes a choice
        with a generic parameters and gives a numeric score used for comparison.
        """
        self.accepted_choices = np.array([])
        self.scores = np.array([])
        self.rejected_choices = np.array([])
        self._choices = choices
        self._filter = filter
        self._criteria = criteria
        self._updated = False
        self._unprocessed_choices = []
        self._filter_parameters = None
        self._criteria_parameters = None

    def optimize(self):
        """
        Performs the optimization process

        It sets accepted_choices, rejected_choices and scores accordingly. This method performs the
        entire process only in the following cases: 1.The first time it is called. 2.Criteria, filter or
        the entire choices list has been setted to a new value. This method performs a partial process
        when new choices are added where only these choices pass through the entire process and then they
        are inserted in the sorted list according to their scores. Otherwise, this method does not do
        anything.
        """
        if self._updated:
            if self._unprocessed_choices == []:
                return
            else:
                accpeted, rejected, scores = self._optimize(self._unprocessed_choices)
                if rejected.shape[0] != 0:
                    self.rejected_choices = np.concatenate(self.rejected_choices, rejected)
                if scores.shape[0] != 0:
                    indices = np.searchsorted(self.scores, scores)
                    self.scores = np.insert(self.scores, indices, scores)
                    self.accepted_choices = np.insert(self.accepted_choices, indices, accpeted)
                self._unprocessed_choices = []
        else:
            self.accepted_choices, self.rejected_choices, self.scores = self._optimize(self._choices)
            self._updated = True
            self._unprocessed_choices = []

    def _optimize(self, choices):
        """
        Performs the optimization process on a list of objects

        It does not affect the member variables accepted_choices, rejected_choices and scores. This method
        performs the entire process on the passed object rather than the member variable choices.
        :param list choices: The list of objects to be sorted by the process
        :return 3-tuble: accepted_choices, rejected_choices and scores. where accepted_choices represents
        a list containing all objects that passed the filter test sorted by their scores descending. and
        rejected_choices represents a list containing all objects that could not pass the filter test.
        scores represents an 1d ndarray containing the scores of the accepted_choices in the same order
        """
        accepted_choices = []
        scores = []
        rejected_choices = []
        for choice in choices:
            print('next: \n')
            if self._filter(choice, self._filter_parameters):
                accepted_choices.append(choice)
                scores.append(self._criteria(choice, self._criteria_parameters))
            else:
                rejected_choices.append(choice)
        scores = np.array(scores)
        accepted_choices = np.array(accepted_choices)
        rejected_choices = np.array(rejected_choices)
        sorted_indices = np.argsort(scores)
        accepted_choices = accepted_choices[sorted_indices]
        scores = scores[sorted_indices]
        return accepted_choices, rejected_choices, scores

    def set_filter(self, filter):
        """
        Sets the filter member variable

        :param function filter: The new value of filter variable
        """
        self._filter = filter
        self._updated = False

    def set_criteria(self, criteria):
        """
        Sets the criteria member variable

        Calling this method between two optimize calls will make the second call to optimize performs the
        entire process.

        :param function criteria: The new value of criteria variable
        """
        self._criteria = criteria
        self._updated = False

    def set_choices(self, choices):
        """
        Sets the choices member variable

        Calling this method between two optimize calls will make the second call to optimize performs the
        entire process.

        :param list choices: The new value of choices variable
        """
        self._choices = choices
        self._updated = False

    def add_choice(self, choice):
        """
        Adds a choice to the choices member variable

        Calling this method between two optimize calls will make the second call to optimize only handles
        the added choices.

        :param object choice: The new choice to be added to the choices variable
        """
        self._choices.append(choice)
        self._unprocessed_choices.append(choice)

    def add_choices(self, choices):
        """
        Adds choices to the choices member variable

        Calling this method between two optimize calls will make the second call to optimize only handles
        the added choices.

        :param list choice: The new choices to be added to the choices variable
        """
        self._choices = self._choices + choices
        self._unprocessed_choices = self._unprocessed_choices + choices

    def set_criteria_parameters(self, parameters):
        """
        sets a parameters variable to used in criteria calls

        Calling this method between two optimize calls will make the second call to optimize performs the
        entire process.

        :param object parameters: The new value of _critera_parameters variable
        """
        self._criteria_parameters = parameters
        self._updated = False

    def set_filter_parameters(self, parameters):
        """
        sets a parameters variable to used in filter calls

        Calling this method between two optimize calls will make the second call to optimize performs the
        entire process.

        :param object parameters: The new value of _filter_parameters variable
        """
        self._filter_parameters = parameters
        self._updated = False