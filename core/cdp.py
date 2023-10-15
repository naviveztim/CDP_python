""" Concatenated Decision Paths (CDP) main class"""

import os.path
import pickle
import csv
from collections import defaultdict
import numpy as np
from utils.utils import try_except, similarity_coefficient
from utils.logger import logger
from utils.dataset import Dataset
from core.shapelet_classifier import ShapeletClassifier


# Filename of trained model - contains sequence of decision trees
MODEL_FILENAME = 'cdp_model.pickle'
# Filename of csv file that contains predicted class indexes
PATTERNS_FILE_NAME = 'patterns.csv'


class CDP:
    """ Concatenated Decision Paths (CDP) method implementation"""

    def __init__(self
                 , dataset: Dataset
                 , model_folder: str
                 , num_classes_per_tree: int
                 , pattern_length: int
                 ):

        self.model_folder = model_folder
        self.num_classes_per_tree = num_classes_per_tree
        self.pattern_length = pattern_length
        self.patterns = []
        self.classification_trees = {}
        self.train_dataset = dataset

    def _load_patterns(self, csv_file_path: str):
        """ Load class indexes along with their representative patterns (LLR..)"""

        # Initialize an empty list to store the loaded data
        self.patterns = []

        # Read the data from the CSV file
        with open(csv_file_path, mode='r', encoding='utf-8') as file:
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
        with open(csv_file_path, mode='w', encoding='utf-8', newline='') as file:
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
        """ Load model- classification trees, along with decision patterns """
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

        logger.info("Training...")

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
        for class_index, time_series in self.train_dataset.iterrows():
            decision_pattern = ''.join([classification_tree.build_classification_path(time_series)
                                       for classification_tree in self.classification_trees.values()]
                                       )
            self.patterns.append((class_index, decision_pattern))

        # Save patterns
        self._save_patterns(os.path.join(self.model_folder, PATTERNS_FILE_NAME))

    @try_except
    def predict(self
                , dataset: Dataset  # pd.DataFrame
                ) -> list:

        """ Predict indexes of given time series datset"""

        logger.info("Predicting...")

        predicted_class_indexes = []

        # Classify by comparing decision patterns
        for _, time_series in dataset.iterrows():

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
