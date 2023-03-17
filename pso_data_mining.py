class ShapeletClassifier:

    def __init__(self, min_length: int
                     , max_length: int
                     ):
        self.min_length = min_length
        self.max_length = max_length
        self.classification_tree_path = ""
        self.best_classification_tree = None
