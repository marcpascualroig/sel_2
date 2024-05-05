import numpy as np
from decision_treev2 import DecisionTreeClassifier
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
        self.feature_frequencies_2 = np.zeros(tot_features)
        self.random_seed = 42

    def fit(self, X, y):
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        if self.n_features != "random" and self.tot_features < self.n_features:
            print('number of features is larger than the total number of features!')
            exit()
        if self.n_features == "random":
            random_num_features = True
        else:
            random_num_features = False

        for _ in range(self.n_estimators):
            if random_num_features:
                self.n_features = random.randint(1, self.tot_features)
            #random subset of features of size F
            selected_features = random.sample(range(self.tot_features), self.n_features)
            #decision tree
            X_aux=X[:, selected_features]
            tree = DecisionTreeClassifier(num_features=self.n_features, selected_feature_indices=selected_features, num_random_features=None, max_depth=self.max_depth)
            freqs1, freqs2 = tree.fit(X_aux, y)
            self.feature_frequencies[selected_features] += freqs1
            self.feature_frequencies_2[selected_features] += freqs2
            self.estimators.append(tree)
        return self.feature_frequencies, self.feature_frequencies_2

    def predict(self, X):
        predictions = np.empty((len(X), len(self.estimators)), dtype=object)
        for i, tree in enumerate(self.estimators):
            predictions[:, i] = tree.predict(X)
        majority_classes = np.empty(len(X), dtype=object)
        for idx in range(len(X)):
            unique_classes, counts = np.unique(predictions[idx], return_counts=True)
            max_count = np.max(counts)
            most_voted_classes = unique_classes[counts == max_count]
            majority_classes[idx] = most_voted_classes[0]
        return majority_classes



