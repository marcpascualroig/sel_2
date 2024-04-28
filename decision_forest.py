import numpy as np
from decision_tree import DecisionTreeClassifier
import random


class DecisionForest:
    def __init__(self, tot_features, NT=1, F=None, max_depth=None):
        self.n_estimators = NT
        self.tot_features = tot_features
        if F is None:
            self.n_features = tot_features
        else:
            self.n_features = F
        self.max_depth = max_depth
        self.estimators = []
        self.feature_frequencies = np.zeros(tot_features)

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            #random subset of features of size F
            selected_features = random.sample(range(self.tot_features), self.n_features)
            print(selected_features)
            #decision tree
            tree = DecisionTreeClassifier(num_features=self.tot_features, num_random_features=None, features_indices= selected_features, max_depth=self.max_depth)
            new_frequencies = tree.fit(X, y)
            self.feature_frequencies += new_frequencies
            self.estimators.append(tree)

    def predict(self, X):
        predictions = np.empty((len(X), len(self.estimators)), dtype=object)
        for i, tree in enumerate(self.estimators):
            predictions[:, i] = tree.predict(X)
            #print(predictions[:, i])
        majority_classes = np.empty(len(X), dtype=object)
        for idx in range(len(X)):
            unique_classes, counts = np.unique(predictions[idx], return_counts=True)
            max_count = np.max(counts)
            most_voted_classes = unique_classes[counts == max_count]
            majority_classes[idx] = random.choice(most_voted_classes)
        #print(majority_classes)
        return majority_classes



