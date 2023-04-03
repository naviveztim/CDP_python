import pandas as pd
import pickle
import os
from collections import Counter
from itertools import chain
import numpy as np

from ShapeletDataMining.shapelet import Shapelet
from ShapeletDataMining.shapelet_classifier import ShapeletClassifier
from PSO.pso import ShapeletsPso
from Utils.utils import from_ucr_txt, subsequent_distance
from Utils.btree import BTree
from itertools import combinations, permutations


class CDP:

    """ Main class in Concatenated Decision Paths (CDP) method implementation"""

    def __init__(self
                 , dataset_filepath: str
                 , classifiers_folder: str
                 , num_classes_per_tree: int
                 , pattern_length: int
                 , compresson_factor: int = None
                 , original_or_derivate: str = None
                 , normalize: bool = False):

        self.classifiers_folder = classifiers_folder
        self.num_classes_per_tree = num_classes_per_tree
        self.pattern_length = pattern_length
        self.compresson_factor = compresson_factor
        self.original_or_derivate = original_or_derivate
        self.normalize = normalize
        self.dataset = from_ucr_txt(dataset_filepath)

        # Apply averaging on time series values
        if compresson_factor:
            self.dataset['values'] = self.dataset['values'].apply(
                lambda x: pd.Series(x).rolling(window=compresson_factor, min_periods=1).mean().tolist())

        # Take original or derivative time series values
        if original_or_derivate and (original_or_derivate == 'D' or original_or_derivate == 'd'):
            self.dataset['values'] = self.dataset['values'] \
                .apply(lambda x: [x[i + 1] - x[i] for i in range(len(x) - 1)])

        # Apply normalization
        if normalize:
            self.dataset['values'] = self.dataset['values'] \
                .apply(lambda x: [(i - np.mean(x)) / np.std(x) for i in x])

    def fit(self):

        shapelet_classifier = ShapeletClassifier(dataset=self.dataset
                                                 , classifiers_folder=self.classifiers_folder
                                                 , num_classes_per_tree=self.num_classes_per_tree
                                                 , pattern_length=self.pattern_length)

        classifiers = shapelet_classifier.create_and_train_classifiers()

        # TODO: Implement ShapeletClassifier's 'classify' method
