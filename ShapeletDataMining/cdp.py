import pandas as pd
import pickle
import os
from collections import Counter
from itertools import chain
import numpy as np
from functools import reduce
from collections import defaultdict

from Utils import utils
from ShapeletDataMining.shapelet import Shapelet
from ShapeletDataMining.shapelet_classifier import ShapeletClassifier
from PSO.pso import ShapeletsPso
from Utils.utils import from_ucr_txt, subsequent_distance
from Utils.btree import BTree
from itertools import combinations, permutations


class CDP:

    """ Main class of Concatenated Decision Paths (CDP) method implementation"""

    def __init__(self
                 , train_dataset_filepath: str
                 , classifiers_folder: str
                 , delimiter: str
                 , num_classes_per_tree: int
                 , pattern_length: int
                 , compression_factor: int = None
                 , original_or_derivate: str = None
                 , normalize: bool = False):

        self.classifiers_folder = classifiers_folder
        self.num_classes_per_tree = num_classes_per_tree
        self.pattern_length = pattern_length
        self.train_dataset = from_ucr_txt(train_dataset_filepath, delimiter)
        self.patterns: list[str] = []
        self.classification_trees: list[BTree] = []
        self._process_train_dataset(compression_factor, normalize, original_or_derivate)

    def _process_train_dataset(self, compression_factor, normalize, original_or_derivate):

        # Apply compression on time series values
        if compression_factor > 1:
            self.train_dataset['values'] = self.train_dataset['values'] \
                .apply(lambda x: pd.Series(x).rolling(window=compression_factor, center=False)
                                             .mean()
                                             .dropna()
                                             .tolist()[::2])

        # Take original/derivative time series values
        if original_or_derivate == 'D' or original_or_derivate == 'd':
            self.train_dataset['values'] = self.train_dataset['values'] \
                .apply(lambda x: [x[i + 1] - x[i] for i in range(len(x) - 1)])

        # Apply normalization
        if normalize == 'Y' or normalize == 'y':
            self.train_dataset['values'] = self.train_dataset['values'] \
                .apply(lambda x: (x - np.mean(x)) / np.std(x))

    def fit(self):

        shapelet_classifier = ShapeletClassifier(dataset=self.train_dataset
                                                 , classifiers_folder=self.classifiers_folder
                                                 , num_classes_per_tree=self.num_classes_per_tree
                                                 , pattern_length=self.pattern_length)

        # Define and train number of specified classification trees
        self.classification_trees = shapelet_classifier.create_and_train_classifiers()

        # Take equal number of samples from every class index
        min_samples = max(10, self.train_dataset.groupby('class_index').size().min())

        # Create patterns collection in format:
        #  [(0, 'LLRLLLLRLLL')
        # , (0, 'LLRRLLLRLLL')
        # , (1, 'RRRLLRRRLLL')
        # , (1, 'RRRRLRRRRLL')]
        for _, time_series in self.train_dataset.sample(min_samples, replace=True).iterrows():
            self.patterns.append((time_series['class_index']
                                  , ''.join([classification_tree.build_classification_path(time_series)
                                             for classification_tree in self.classification_trees])))
        # TEST
        for pattern in self.patterns:
            print(f'Index: {pattern[0]}, Pattern: {pattern[1]}')

    def predict(self, test_dataset_filepath: str, delimiter: str = ',') -> list:

        """ Predict indexes of given time series """

        # Read test dataset
        test_dataset = from_ucr_txt(test_dataset_filepath, delimiter)

        # Apply pre-processing, already applied to train dataset
        test_dataset = self.process_dataset(test_dataset)

        predicted_class_indexes = []

        # Classify by comparing decision patterns
        for _, time_series in test_dataset.iterrows():

            # Find the pattern for given time series
            pattern = ''.join(self.classification_trees.build_classification_path(time_series))

            # Find similarity between found pattern and saved during training
            similarities = [(i, utils.similarity_coeff(pattern, s)) for i, s in self.patterns]

            # Sort tuples by closest distance and take 10 closest
            biggest_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:10]

            # Find most popular index among the closest distances
            freq_dict = defaultdict(int)
            for tup in biggest_similarities:
                freq_dict[tup[0]] += 1

            # Finding index with maximum frequency
            predicted_class_indexes.append(max(freq_dict, key=freq_dict.get))

        return predicted_class_indexes






