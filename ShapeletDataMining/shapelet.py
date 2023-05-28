import numpy as np
import sys
from math import isclose


class Shapelet:
    def __init__(self, values, optimal_split_distance, best_information_gain):
        self.values = values
        self.optimal_split_distance = optimal_split_distance
        self.best_information_gain = best_information_gain
        self._left_class_index = -1
        self._right_class_index = -1

    @property
    def left_class_index(self):
        return self._left_class_index

    @left_class_index.setter
    def left_class_index(self, value):
        if value is not None and type(value) is not int:
            raise Exception('Left class index is incorrect!')
        self._left_class_index = value

    @property
    def right_class_index(self):
        return self._right_class_index

    @right_class_index.setter
    def right_class_index(self, value):
        if value is not None and type(value) is not int:
            raise Exception('Right class index is incorrect!')
        self._right_class_index = value

    def __eq__(self, other):
        if not isinstance(other, Shapelet):
            raise Exception("Not correct Shapelet instance!")

        return \
            np.array_equiv(self.values, other.values)\
            and isclose(self.optimal_split_distance, other.optimal_split_distance)\
            and isclose(self.best_information_gain, other.best_information_gain)\
            and self.left_class_index == other.left_class_index\
            and self.right_class_index == other.right_class_index

    def compare(self, other):

        if not isinstance(other, Shapelet):
            raise Exception("Not correct Shapelet instance!")

        if self == other:
            return 0
        elif self.left_class_index == other.left_class_index\
             or self.left_class_index == other.right_class_index:
            return -1
        elif self.right_class_index == other.left_class_index\
             or self.right_class_index == other.right_class_index:
            return 1
        else:
            return -2

    def __repr__(self):

        print(f'values: {self.values}')
        print(f'optimal_split_distance: {self.optimal_split_distance}')
        print(f'best_information_gain: {self.best_information_gain}')
        print(f'left_class_index: {self.left_class_index}')
        print(f'right_class_index: {self.right_class_index}')
