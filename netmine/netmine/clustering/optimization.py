import numpy as np
import pandas as pd

class Optimizer:
    """
    A class that performs general optimization

    todo: list should be changed to numpy
    todo: filter and criteria have to have 2 arguments each
    todo: add store and load to documentation
    Attributes
    ----------
    accepted_choices : list
        A list containing all objects that passed the filter test sorted by their scores descending
    rejected_choices : list
        A list containing all objects that could not pass the filter test
    scores : ndarray
        An 1d ndarray containing the scores of the accepted_choices in the same order
    """

    def __init__(self, choices=[], filter=lambda x,y: True, criteria=lambda x,y: 0, store=None):
        """
        :param list choices: The competing objects.
        :param function filter: A filter used to exclude some of the choices. It is a function that takes
        a choice with a generic parameter and gives a boolean value which specify whether to include the
        choice or not. If True, the choice is included.
        :param function criteria: The criteria used for optimization. It is a function that takes a choice
        with a generic parameters and gives a numeric score used for comparison.
        """
        self.accepted_choices = np.array([]).astype(np.object)
        self.scores = np.array([])
        self.rejected_choices = np.array([]).astype(np.object)
        self._choices = choices
        self._filter = filter
        self._criteria = criteria
        self._updated = False
        self._unprocessed_choices = []
        self._filter_parameters = None
        self._criteria_parameters = None
        self._store = store

    def from_io(self, load, filename):
        return load(filename)

    def __str__(self):
        string = ''
        for i,choice in enumerate(self.accepted_choices):
            string = string + str(i) + ' : ' + str(choice) + ' ---- score : ' + str(self.scores[i]) + '\n'
        return string

    def optimize(self, store_filename=None, verbose=False):
        """
        Performs the optimization process

        It sets accepted_choices, rejected_choices and scores accordingly. This method performs the
        entire process only in the following cases: 1.The first time it is called. 2.Criteria, filter or
        the entire choices list has been setted to a new value. This method performs a partial process
        when new choices are added where only these choices pass through the entire process and then they
        are inserted in the sorted list according to their scores. Otherwise, this method does not do
        anything.
        """
        if not self._updated:
            self._unprocessed_choices = self._choices.copy()
            self.accepted_choices = np.array([]).astype(np.object)
            self.rejected_choices = np.array([]).astype(np.object)
            self.scores = np.array([])
            self._updated = True
        choices_buffer = self._unprocessed_choices.copy()
        for i,choice in enumerate(choices_buffer):
            if verbose:
                print(str(i) + ' : ')
            if self._filter(choice, self._filter_parameters):
                score = self._criteria(choice, self._criteria_parameters)
                i = np.searchsorted(self.scores, score)
                self.scores = np.insert(self.scores, i, score)
                self.accepted_choices = np.insert(self.accepted_choices, i, choice)
            else:
                self.rejected_choices = np.append(self.rejected_choices, choice)
            self._unprocessed_choices.pop(0)
            if store_filename != None:
                self.store_self(store_filename)

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

    def store_self(self, filename):
        self._store(self, filename)