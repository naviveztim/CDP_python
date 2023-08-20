import pandas as pd
import os
from collections import Counter
from itertools import chain
import numpy as np
from core.shapelet import Shapelet
from core.pso import ShapeletsPso
from utils.utils import subsequent_distance
from utils.btree import BTree
from itertools import combinations, permutations
from utils.logger import logger


class ShapeletClassifier:

    def __init__(self
                 , dataset: pd.DataFrame
                 , classifiers_folder: str
                 , num_classes_per_tree: int
                 , pattern_length: int):

        self.classifiers_folder = classifiers_folder
        self.num_classes_per_tree = num_classes_per_tree
        self.pattern_length = pattern_length
        self.balanced_dataset = pd.DataFrame()
        if dataset is not None and not dataset.empty:
            for i in dataset['class_index'].unique():
                self.balanced_dataset = \
                    pd.concat([self.balanced_dataset,
                               dataset.loc[dataset['class_index'] == i].sample(10, replace=True)
                               ])
        self.balanced_dataset = self.balanced_dataset.reset_index().drop(columns=['index'])

    @staticmethod
    def _build_tree(permutation: tuple):
        """ Build tree from given permutation """

        # Initialize the decision tree
        tree = BTree(permutation[0])

        # Build the tree
        for p in permutation:
            tree.add(tree.root, p)

        return tree

    @staticmethod
    def _count_left_right(shapelet: Shapelet, class_dataset: pd.DataFrame):

        # Take number of time series which distance to the shapelet is less than optimal split distance
        count_left = 0
        for values in class_dataset['values']:
            if subsequent_distance(values, shapelet.values) < shapelet.optimal_split_distance:
                count_left += 1

        return count_left, class_dataset.shape[0] - count_left

    def _split_classes(self, shapelet: Shapelet, class_index_a: int, class_index_b: int, dataset: pd.DataFrame):

        """ Assign left and right class index to given shapelet"""

        class_dataset_a = dataset[dataset.class_index == class_index_a]
        count_left_a, count_right_a = self._count_left_right(shapelet, class_dataset_a)

        class_dataset_b = dataset[dataset.class_index == class_index_b]
        count_left_b, count_right_b = self._count_left_right(shapelet, class_dataset_b)

        # Assign class index, for which majority of samples have
        shapelet.left_class_index = class_index_a if count_left_a > count_left_b else class_index_b
        shapelet.right_class_index = class_index_a if count_right_a > count_right_b else class_index_b

    def _find_shapelet(self, class_index_a: int, class_index_b: int) -> Shapelet:
        """ Find shapelet and its parameters such as optimal split distance, left and right class index"""

        train_dataset = self.balanced_dataset[self.balanced_dataset['class_index']
                                                  .isin([class_index_a, class_index_b])]
        # Start PSO algorithm to find shapelet that separates two classes
        min_length = 3
        max_length = min(len(x) for x in train_dataset['values'])
        step = (max_length - min_length)//20
        step = step if step > 0 else 1
        num_classes = len(self.balanced_dataset['class_index'].unique())
        step = step if num_classes >= 4 else 1
        min_train_value = np.min(train_dataset.iloc[0]['values'])
        max_train_value = np.max(train_dataset.iloc[0]['values'])
        shapelet_pso = ShapeletsPso(min_length=min_length
                                    , max_length=max_length
                                    , step=step
                                    , min_position=min_train_value
                                    , max_position=max_train_value
                                    , min_velocity=min_train_value
                                    , max_velocity=max_train_value
                                    , train_dataframe=train_dataset)
        shapelet_pso.start_pso()

        # Fill shapelet parameters- shapelet values, the best info gain, optimal split distance
        shapelet = Shapelet(values=shapelet_pso.best_particle.position[:shapelet_pso.best_particle.length]
                            , best_information_gain=shapelet_pso.best_particle.best_information_gain
                            , optimal_split_distance=shapelet_pso.best_particle.optimal_split_distance)

        # Fill shapelet parameters-left and right class indexes of the shapelet
        self._split_classes(shapelet, class_index_a, class_index_b, train_dataset)

        return shapelet

    def _get_required_classificators_names(self, group: list):

        required_classificators_names = {}
        for combination in group:

            # Check if classifier with such name already exists
            base_name = os.path.join(
                self.classifiers_folder
                , "Classificatin_tree_" + "_".join([str(x) for x in combination]))
            extension = '.pickle'

            # Make unique filename
            counter = 1
            unique_filename = base_name + extension
            while unique_filename in required_classificators_names:
                unique_filename = f"{base_name}_({counter}){extension}"
                counter += 1

            required_classificators_names[unique_filename] = combination

        return required_classificators_names

    def _test_tree_accuracy(self, tree: BTree, classes_in_combination: tuple):

        """ Finds average accuracy of given classification tree
            """

        for class_index in classes_in_combination:
            # Average accuracy
            acc = 0.0

            # Extract the time series with given class index
            train_dataset = \
                self.balanced_dataset.loc[self.balanced_dataset['class_index'] == class_index].sample(n=10)

            # Number of correctly classified time series for given index
            num_correctly_classified = 0

            for _, time_series in train_dataset.iterrows():
                current_node = tree.root

                # Iterate the tree
                while True:
                    if current_node is None:
                        break
                    distance = subsequent_distance(time_series["values"], current_node.shapelet.values)

                    # Left wing
                    if distance <= current_node.shapelet.optimal_split_distance:
                        if current_node.left is not None:
                            current_node = current_node.left
                            continue
                        if current_node.shapelet.left_class_index == class_index:
                            num_correctly_classified += 1
                        break

                    # Right wing
                    if distance > current_node.shapelet.optimal_split_distance:
                        if current_node.right is not None:
                            current_node = current_node.right
                            continue
                        if current_node.shapelet.right_class_index == class_index:
                            num_correctly_classified += 1
                        break

            acc += num_correctly_classified/train_dataset.shape[0]

        return acc

    def _find_most_accurate_tree(self, shapelets: list, classes_in_combination: tuple):

        """ Tries variety of combinations to build most accurate tree"""

        best_acc = 0.0
        best_tree: BTree = None

        for combination in list(combinations(shapelets, len(classes_in_combination)-1)):
            for permutation in list(permutations(combination)):
                tree = self._build_tree(permutation)
                acc = self._test_tree_accuracy(tree, classes_in_combination)
                if best_acc < acc:
                    best_acc = acc
                    best_tree = tree
                    best_tree.accuracy = acc

        return best_tree

    def _create_group(self) -> list:

        class_indexes = self.balanced_dataset['class_index'].unique()
        num_class_indexes = len(class_indexes)
        num_allowed_indexes = self.pattern_length * self.num_classes_per_tree // num_class_indexes

        # Generate all possible combinations of 3 classes from the range 0-15
        combs = list(combinations(class_indexes, self.num_classes_per_tree))

        # Filter the combinations to only keep those where each class appears equally
        valid_combs = []
        num_iterations = 0
        # The max number of iterations is set to 2000 artificially as in some cases
        # the output cannot contain all the numbers uniformly distributed and the cycle never ends.
        while not valid_combs \
            or any(freq[class_] < num_allowed_indexes for class_ in list(chain(*valid_combs)))\
                and num_iterations < 2000:
            num_iterations += 1
            for comb in combs:
                freq = Counter(list(chain(*valid_combs)))
                if not valid_combs:
                    valid_combs.append(comb)
                else:
                    if all(freq[class_] < num_allowed_indexes for class_ in comb)\
                            and any(freq[class_] == min(freq.values()) for class_ in comb):
                        valid_combs.append(comb)
                # The numbers in combinations can  be almost uniformly distributed-
                # aka one or two numbers might differ

        return valid_combs

    def _create_and_train_tree(self, classes_in_combination: tuple) -> BTree:
        """ Find shapelets for every pair of classes in given combination. Build the classification
        tree and serialize the most accurate tree"""
        shapelets = []

        # Split combination of classes to all possible two classes combinations
        two_classes_combinations = list(combinations(classes_in_combination, 2))

        # Collect shapelets that split combinations of two classes
        for c in two_classes_combinations:
            class_index_a, class_index_b = c
            shapelet = self._find_shapelet(class_index_a, class_index_b)

            # Find shapelet that split the two classes
            if shapelet.right_class_index is not None\
               and shapelet.left_class_index is not None\
               and shapelet.right_class_index != shapelet.left_class_index:
                shapelets.append(shapelet)

        # Find tree, that gives the best accuracy for given combination of class indexes
        best_tree = self._find_most_accurate_tree(shapelets, classes_in_combination)

        return best_tree

    def create_and_train_classifiers(self, classification_trees: dict):

        """ Create, train group of classifiers defined by input arguments """

        # Find needed combination of classes, defined by input arguments
        group = self._create_group()
        logger.debug(group)

        # Get newly required classifiers- filename: class combination
        required_trees = self._get_required_classificators_names(group)

        # Define trees to be created
        classifiers_file_names = [x for x in required_trees if x not in classification_trees]

        # Create classification tree
        for i, classifier_file_name in enumerate(classifiers_file_names):
            if classifier_file_name:
                classification_trees[classifier_file_name] = \
                    self._create_and_train_tree(classes_in_combination=required_trees[classifier_file_name])
                # Training progress
                logger.info(f"Training classifier {i+1}/{len(classifiers_file_names)}")















