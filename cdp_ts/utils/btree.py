from cdp_ts.utils.utils import subsequent_distance
from cdp_ts.core.shapelet import Shapelet
import numpy as np


class BTree:

    class Node:
        def __init__(self, shapelet: Shapelet):
            self.shapelet: Shapelet = shapelet
            self.left = None
            self.right = None
            self.depth = 0

    def __init__(self, shapelet: Shapelet):
        self.root: BTree.Node = self.Node(shapelet)
        self.accuracy: float = 0.0
        self.num_nodes = 1

    def build_classification_pattern(self, time_series: np.array) -> str:

        """ Build classification pattern that represent given time series"""
        current_node: BTree.Node = self.root
        path_string: str = ''

        # Iterate the tree
        while True:
            if current_node is None:
                break

            distance = subsequent_distance(time_series
                                           , current_node.shapelet.values)

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

        # Pad the answer
        while len(path_string) < self.num_nodes:
            path_string += "0"
        return path_string

    def add(self, node: Node, shapelet: Shapelet) -> bool:
        """ Add new node to the tree"""

        add_result: bool = False

        if node is None or shapelet is None:
            return False

        compare_result = node.shapelet.compare(shapelet)

        if compare_result == 0:
            return True

        elif compare_result == -1:
            if node.left is None:
                node.left = BTree.Node(shapelet)
                node.depth += 1
            self.num_nodes += 1
            return self.add(node.left, shapelet)

        elif compare_result == 1:
            if node.right is None:
                node.right = BTree.Node(shapelet)
                node.depth += 1
            self.num_nodes += 1
            return self.add(node.right, shapelet)

        elif compare_result == -2:
            add_result = self.add(node.left, shapelet)
            if not compare_result:
                self.num_nodes += 1
                add_result = self.add(node.right, shapelet)

        return add_result





