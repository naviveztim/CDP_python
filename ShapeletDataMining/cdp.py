import pandas as pd
import numpy as np
from collections import defaultdict
from Utils import utils
from Utils.logger import logger
from ShapeletDataMining.shapelet_classifier import ShapeletClassifier
from Utils.btree import BTree


class CDP:

    """ Main class of Concatenated Decision Paths (CDP) method implementation"""

    def __init__(self
                 , dataset: pd.DataFrame
                 , classifiers_folder: str
                 , num_classes_per_tree: int
                 , pattern_length: int
                 , compression_factor: int = None
                 , original_or_derivate: str = None
                 , normalize: bool = False):

        self.classifiers_folder = classifiers_folder
        self.num_classes_per_tree = num_classes_per_tree
        self.pattern_length = pattern_length
        self.train_dataset = dataset
        self.patterns: list[str] = []
        self.classification_trees: list[BTree] = []
        self.compression_factor = compression_factor
        self.normalize = normalize
        self.original_or_derivate = original_or_derivate
        self._process_dataset(self.train_dataset)

    def _process_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:

        # Apply compression on time series values
        if self.compression_factor > 1:
            dataset['values'] = dataset['values'] \
                .apply(lambda x: pd.Series(x).rolling(window=self.compression_factor, center=False)
                                             .mean()
                                             .dropna()
                                             .tolist()[::2])

        # Take original/derivative time series values
        if self.original_or_derivate == 'D' or self.original_or_derivate == 'd':
            dataset['values'] = dataset['values'] \
                .apply(lambda x: [x[i + 1] - x[i] for i in range(len(x) - 1)])

        # Apply normalization
        if self.normalize == 'Y' or self.normalize == 'y':
            dataset['values'] = dataset['values'] \
                .apply(lambda x: (x - np.mean(x)) / np.std(x))

    def fit(self):

        shapelet_classifier = ShapeletClassifier(dataset=self.train_dataset
                                                 , classifiers_folder=self.classifiers_folder
                                                 , num_classes_per_tree=self.num_classes_per_tree
                                                 , pattern_length=self.pattern_length)

        # Define and train number of specified classification trees
        self.classification_trees = shapelet_classifier.create_and_train_classifiers()

        # Take equal number of samples from every class index
        #min_samples = max(10, self.train_dataset.groupby('class_index').size().min())

        # Create patterns collection in format:
        #  [(0, 'LLRLL...LLRLLL')
        # , (0, 'LLRRL...LLRLLL')
        # , (1, 'RRRLL...RRRLLL')
        # , (1, 'RRRRL...RRRRLL')]
        for _, time_series in self.train_dataset.iterrows():
            self.patterns.append((time_series['class_index']
                                  , ''.join([classification_tree.build_classification_path(time_series)
                                             for classification_tree in self.classification_trees])))
        # Sanity check
        for pattern in self.patterns:
            logger.info(f'Index: {pattern[0]}, Pattern: {pattern[1]}')

    def predict(self, dataset: pd.DataFrame) -> list:

        """ Predict indexes of given time series """

        # Apply pre-processing, already applied to train dataset
        self._process_dataset(dataset)

        predicted_class_indexes = []

        # Classify by comparing decision patterns
        for _, time_series in dataset.iterrows():

            # Find the pattern for given time series
            pattern = ''.join([classification_tree.build_classification_path(time_series)
                               for classification_tree in self.classification_trees])
                      # ''.join(self.classification_trees.build_classification_path(time_series))

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






