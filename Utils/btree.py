import numpy as np
from ShapeletDataMining.shapelet import Shapelet


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





