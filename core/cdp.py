import os.path
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
from utils.utils import try_except, similarity_coefficient
from utils.logger import logger
from core.shapelet_classifier import ShapeletClassifier
import csv

# Filename of trained model - contains sequence of decision trees
MODEL_FILENAME = 'cdp_model.pickle'
# Filename of csv file that contains predicted class indexes
PATTERNS_FILE_NAME = 'patterns.csv'


class CDP:
    """ Concatenated Decision Paths (CDP) method implementation"""

    def __init__(self
                 , dataset: pd.DataFrame
                 , model_folder: str
                 , num_classes_per_tree: int
                 , pattern_length: int
                 , compression_factor: int = None
                 , derivative: bool = False
                 , normalize: bool = False):

        self.model_folder = model_folder
        self.num_classes_per_tree = num_classes_per_tree
        self.pattern_length = pattern_length
        self.patterns: list[tuple] = []
        self.classification_trees = dict()
        self.compression_factor = compression_factor
        self.normalize = normalize
        self.derivative = derivative
        self.train_dataset = self._process_dataset(dataset)

    def _process_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """ Process given dataframe with compression, normalization or extract derivative """

        dataset = dataset.copy()

        if dataset is None or dataset.empty:
            return pd.DataFrame()

        # Apply compression on time series
        if self.compression_factor > 1:
            dataset['values'] = dataset['values'] \
                .apply(lambda x: pd.Series(x).rolling(window=self.compression_factor, center=False)
                                             .mean()
                                             .dropna()
                                             .tolist()[::2])

        # Extract derivative from time series
        if self.derivative:
            dataset['values'] = dataset['values'] \
                .apply(lambda x: [x[i + 1] - x[i] for i in range(len(x) - 1)])

        # Apply normalization on time series
        if self.normalize:
            dataset['values'] = dataset['values'] \
                .apply(lambda x: (x - np.mean(x)) / np.std(x))

        return dataset

    def _load_patterns(self, csv_file_path: str):
        """ Load class indexes along with their representative patterns (LLR..)"""

        # Initialize an empty list to store the loaded data
        self.patterns = []

        # Read the data from the CSV file
        with open(csv_file_path, mode='r') as file:
            reader = csv.DictReader(file)

            # Assuming the column headers are 'class_index' and 'class_pattern'
            for row in reader:
                class_index = int(row['class_index'])
                class_pattern = row['class_pattern']
                self.patterns.append((class_index, class_pattern))

    def _save_patterns(self, csv_file_path: str):
        """ Save class indexes along with their representative patterns (LLR..) """

        # Load patterns from file
        headers = ['class_index', 'class_pattern']

        # Write the data to the CSV file
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)

            # Write the header row
            writer.writerow(headers)

            # Write the data rows
            for row in self.patterns:
                writer.writerow(row)

    def _save_classification_trees(self, model_folder_path: str):
        """ Save model with classification tree sequence """

        with open(os.path.join(model_folder_path, MODEL_FILENAME), 'wb') as file:
            pickle.dump(self.classification_trees, file)

    def _load_classification_trees(self, model_folder_path: str) -> dict:
        """ Read model file with decision trees sequence """

        try:
            with open(os.path.join(model_folder_path, MODEL_FILENAME), 'rb') as file:
                self.classification_trees = pickle.load(file)
        except Exception as e:
            self.classification_trees = dict()

    @try_except
    def load_model(self):

        # Load model
        self._load_classification_trees(self.model_folder)

        # Load saved patterns
        self._load_patterns(os.path.join(self.model_folder, PATTERNS_FILE_NAME))

        # Sanity check
        for pattern in self.patterns:
           logger.info(f'Index: {pattern[0]}, Pattern: {pattern[1]}')

    @try_except
    def fit(self):

        """ Fills the dictionary with classification trees. If model exists, tries to reuse
        trees if compatible with input arguments requirements"""

        logger.info(f"Training...")

        shapelet_classifier = ShapeletClassifier(dataset=self.train_dataset
                                                 , classifiers_folder=self.model_folder
                                                 , num_classes_per_tree=self.num_classes_per_tree
                                                 , pattern_length=self.pattern_length)

        # Load existing classifiers
        self._load_classification_trees(self.model_folder)

        # Add new classifiers, if required
        shapelet_classifier.create_and_train_classifiers(self.classification_trees)

        # Save all classifiers
        self._save_classification_trees(self.model_folder)

        # Create patterns in format:
        #  [(0, 'LLRLL...LLRLLL')
        # ....
        # , (1, 'RRRRL...RRRRLL')]
        for _, time_series in self.train_dataset.iterrows():
            self.patterns.append((time_series['class_index']
                                  , ''.join([classification_tree.build_classification_path(time_series)
                                             for classification_tree in self.classification_trees.values()])))

        # Save patterns
        self._save_patterns(os.path.join(self.model_folder, PATTERNS_FILE_NAME))

    @try_except
    def predict(self, dataset: pd.DataFrame) -> list:

        """ Predict indexes of given time series datset"""

        logger.info(f"Predicting...")

        # Apply pre-processing, already applied to train dataset
        processed_dataset = self._process_dataset(dataset)

        predicted_class_indexes = []

        # Classify by comparing decision patterns
        for _, time_series in processed_dataset.iterrows():

            # Find the pattern for given time series
            pattern = ''.join([classification_tree.build_classification_path(time_series)
                               for classification_tree in self.classification_trees.values()])

            # Find similarity between found pattern and saved during training
            similarities = [(i, similarity_coefficient(pattern, s)) for i, s in self.patterns]

            # Sort tuples by closest distance and take 10 closest
            biggest_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:10]

            # Find most popular index among the closest distances
            freq_dict = defaultdict(int)
            for tup in biggest_similarities:
                freq_dict[tup[0]] += 1

            # Finding index with maximum frequency
            predicted_class_indexes.append(max(freq_dict, key=freq_dict.get))

        return predicted_class_indexes






