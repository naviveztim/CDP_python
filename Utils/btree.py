import numpy as np
from ShapeletDataMining.shapelet import Shapelet
import pandas as pd
from Utils.utils import subsequent_distance


class BTree:

    class Node:
        def __init__(self, shapelet):
            self.shapelet = shapelet
            self.right = None
            self.left = None
            self.depth = 0

    def __init__(self, shapelet):
        self.root = self.Node(shapelet)
        self.accuracy = 0.0

    def build_classification_path(self, time_series: pd.Series) -> str:

        current_node = self.root
        path_string = ''

        # Iterate the tree
        while True:
            if current_node is None:
                break

            distance = subsequent_distance(time_series["values"], current_node.shapelet.values)

            # Left wing
            if distance <= current_node.shapelet.optimal_split_distance:
                path_string += 'L'
                if current_node.left is not None:
                    current_node = current_node.left
                    continue
                else:
                    break
            # Right wing
            else:
                path_string += 'R'
                if current_node.right is not None:
                    current_node = current_node.right
                    continue
                else:
                    break

        return path_string

    def add(self, node: Node, shapelet: Shapelet)->bool:
        """ Add new node to the tree"""

        add_result = False

        if node is None or shapelet is None:
            return False

        compare_result = node.shapelet.compare(shapelet)

        if compare_result == 0:
            return True

        elif compare_result == -1:
            if node.left is None:
                node.left = BTree.Node(shapelet)
                node.depth += 1
            return self.add(node.left, shapelet)

        elif compare_result == 1:
            if node.right is None:
                node.right = BTree.Node(shapelet)
                node.depth += 1
            return self.add(node.right, shapelet)

        elif compare_result == -2:
            add_result = self.add(node.left, shapelet)
            if not compare_result:
                add_result = self.add(node.right, shapelet)

        return add_result





