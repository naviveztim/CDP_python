import pandas as pd
import pickle
import os
from ShapeletDataMining.shapelet import Shapelet
from PSO.pso import ShapeletsPso
from Utils.utils import from_ucr_txt, subsequent_distance
from Utils.btree import BTree
from itertools import combinations, permutations


class ShapeletClassifier:

    def __init__(self, dataset_filepath: str, classifiers_folder: str):

        self.dataset = from_ucr_txt(dataset_filepath)
        self.classifiers_folder = classifiers_folder

        # Take randomly 10 samples from each class index- faster to train and balanced dataset
        self.balanced_dataset = pd.DataFrame()
        for i in self.dataset['class_index'].unique():
            self.balanced_dataset = pd.concat([self.balanced_dataset
                                              , self.dataset.loc[self.dataset['class_index'] == i]
                                              .sample(5, replace=True)])
        self.balanced_dataset = self.balanced_dataset.reset_index().drop(columns=['index'])

    @staticmethod
    def _build_tree(permutation: tuple):
        """ Build tree from given permutation """

        tree = BTree(permutation[0][0])
        for p in permutation:
            tree.add(p)

        return tree

    @staticmethod
    def _split_classes(shapelet: Shapelet, class_index: int, dataset: pd.DataFrame):

        """ Assign left and right class index to given shapelet"""

        # Take number of time series which distance to the shapelet is less than optimal split distance
        count_less = (subsequent_distance(dataset.values, shapelet.values)
                      < shapelet.optimal_split_distance).sum()

        # Take the rest of the counts
        count_other = dataset.shape[0] - count_less

        # Assign class index, for which majority of samples have
        shapelet.left_class_index = class_index if count_less > count_other else None
        shapelet.right_class_index = class_index if count_less < count_other else None

    def _find_shapelet(self, class_index_a: int, class_index_b: int) -> Shapelet:
        """ Find shapelet and its parameters such as optimal split distance, left and right class index"""

        # Start PSO algorithm to find shapelet that separates two classes
        min_length = 3
        max_length = len(self.balanced_dataset['values'][0])
        step = (max_length - min_length)//20
        step = step if step > 0 else 1
        min_train_value = min(self.balanced_dataset['values'].explode())
        max_train_value = max(self.balanced_dataset['values'].explode())
        shapelet_pso = ShapeletsPso(min_length=min_length
                                    , max_length=max_length
                                    , step=step
                                    , min_position=min_train_value
                                    , max_position=max_train_value
                                    , min_velocity=min_train_value
                                    , max_velocity=max_train_value
                                    , train_dataframe=self.balanced_dataset)
        shapelet_pso.start_pso()

        # Fill shapelet parameters- shapelet values, best info gain, optimal split distance
        shapelet = Shapelet(values=shapelet_pso.best_particle.best_position# TODO: Check if position or best_position
                            , best_information_gain=shapelet_pso.best_particle.best_information_gain
                            , optimal_split_distance=shapelet_pso.best_particle.optimal_split_distance)

        # TODO: Check if only one call of below function is enough
        # Fill shapelet parameters- left and right class indexes of the shapelet
        ShapeletClassifier._split_classes(shapelet, class_index_a, self.balanced_dataset[self.balanced_dataset.class_index == class_index_a])
        ShapeletClassifier._split_classes(shapelet, class_index_b, self.balanced_dataset[self.balanced_dataset.class_index == class_index_b])

        return shapelet

    def _get_serialize_name(self, classes_in_combination: list):

        """ Build filename for serialization """

        # Create classifier filename
        classifier_file_name = os.path.join(
            self.classifiers_folder
            , "Classificatin_tree_" + "_".join([str(x) for x in classes_in_combination])
              + ".pickle")

        # Check if the file already exists, if so add a number to the end of the file name
        i = 1
        while os.path.isfile(classifier_file_name):
            classifier_file_name = f"{os.path.splitext(classifier_file_name)[0]}_{i}.pickle"
            i += 1

        return classifier_file_name

    def _serialize_tree(self, tree: BTree, classes_in_combination: list):

        """ Serialize best tree into classification folder"""

        classifier_file_name = self._get_serialize_name(classes_in_combination)

        # Write the serialized classifier to a file
        with open(classifier_file_name, 'wb') as file:
            pickle.dumps(tree, file)

    def _test_tree_accuracy(self, tree: BTree, classes_in_combination: list):

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

    def _find_most_accurate_tree(self, shapelets: list, classes_in_combination: list):

        """ Tries variety of combinations to build most accurate tree"""

        best_acc = 0.0
        best_tree: BTree = None

        for combination in list(combinations(shapelets, len(classes_in_combination)-1)):
            for permutation in list(permutations(combination)):
                tree = self._build_tree(permutation)
                acc = self._test_tree_accuracy(self, tree, classes_in_combination)
                if best_acc < acc:
                    best_acc = acc
                    best_tree = tree
                    best_tree.accuracy = acc

        return best_tree

    def load_tree(self, classes_in_combination: list) -> BTree:
        """ Deserialize classification tree from disk """
        classifier_file_name = self._get_serialize_name(classes_in_combination)

        try:
            # Read serialized classifier from file
            with open(classifier_file_name, 'rb') as file:
                serialized_tree = pickle.load(file)
        except FileNotFoundError:
            return None

        return serialized_tree

    def train_tree(self, classes_in_combination: list) -> None:
        """ Find shapelets for every pair of classes in given combination. Build the classification
        tree and serialize the most accurate tree"""
        shapelets = []
        combination_indexes = [(x, y) for x in classes_in_combination for y in classes_in_combination if x < y]

        # Collect shapelets that split combinations of two classes
        for c in combination_indexes:
            class_index_a, class_index_b = c
            shapelet = self._find_shapelet(class_index_a, class_index_b)

            # Find shapelet that split the two classes
            if shapelet.right_class_index is not None\
               and shapelet.left_class_index is not None\
               and shapelet.right_class_index != shapelet.left_class_index:
                shapelets.append(shapelet)

        # Find tree, that gives highest accuracy for given combination of class indexes
        best_tree = self._find_most_accurate_tree(shapelets, classes_in_combination)

        # Serialize tree to disk
        self._serialize_tree(best_tree, classes_in_combination)












