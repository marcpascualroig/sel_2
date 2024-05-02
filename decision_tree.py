import numpy as np
from itertools import combinations
import math
import random
class DecisionTreeClassifier:
    def __init__(self, num_features, num_random_features=None, features_indices=None, max_depth=None, min_samples_split=1, min_impurity=-1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        if features_indices is None:
            self.selected_features_indices = range(num_features)
        else:
            self.selected_features_indices = features_indices
        if num_random_features is None or num_random_features<0:
            self.num_random_features = 0
        else:
            self.num_random_features = num_random_features
        self.tree = None
        self.feature_frequencies = np.zeros(num_features)
        self.feature_frequencies_2 = np.zeros(num_features)


    def fit(self, X, y):
        self.tree = self._build_tree(X, y)
        return self.feature_frequencies, self.feature_frequencies_2

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        unique_classes = np.unique(y)

        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or num_samples < self.min_samples_split or len(unique_classes) == 1:
            return {'class': self._most_common_class(y)}

        # Find best split (impurity)
        best_split = self._find_best_split(X, y)
        #print(best_split)

        #stopping criteria, impurity threshold
        if len(best_split)==0 or best_split['impurity'] <= self.min_impurity:
            return {'class': self._most_common_class(y)}

        #add counting for feature importance
        self.feature_frequencies[self.selected_features_indices.index(best_split['feature_index'])] += 1
        self.feature_frequencies_2[self.selected_features_indices.index(best_split['feature_index'])] += num_samples

        # expand tree with the best split
        left_subtree = self._build_tree(*best_split['left'], depth + 1)
        right_subtree = self._build_tree(*best_split['right'], depth + 1)

        return {'feature_index': best_split['feature_index'],
                'partition': best_split['values_subsets'],
                'left': left_subtree,
                'right': right_subtree}

    def _find_best_split(self, X, y):
        num_samples, num_features = X.shape
        best_split = {}
        best_impurity = 1

        #random selection of features
        if self.num_random_features >0:
            features = random.sample(range(num_features), self.num_random_features)
        else:
            features = range(num_features)

        #iterate for all features
        for feature_index in features:
            feature_values = np.unique(X[:, feature_index].astype(str))
            #generate all possible split-point subsets
            partitions = self._get_combinations(feature_values)
            #loop for all possible subsets
            for partition in partitions:
                left_values = partition[0]
                right_values = partition[1]

                left_indices = np.where(np.isin(X[:, feature_index], list(left_values)))
                right_indices = np.where(np.isin(X[:, feature_index], list(right_values)))

                if len(left_indices[0]) < self.min_samples_split or len(right_indices[0]) < self.min_samples_split:
                    continue

                left_impurity = self._calculate_impurity(y[left_indices])
                right_impurity = self._calculate_impurity(y[right_indices])
                impurity = (len(left_indices[0]) * left_impurity + len(right_indices[0]) * right_impurity) / num_samples

                if impurity < best_impurity:
                    best_split = {
                        'feature_index': self.selected_features_indices[feature_index],
                        'values_subsets': partition,
                        'left': (X[left_indices], y[left_indices]),
                        'right': (X[right_indices], y[right_indices]),
                        'impurity': impurity
                    }
                    best_impurity = impurity

        return best_split


    def _get_combinations(self, lst):
        n = len(lst)
        if n % 2 != 0:
            all_combinations = [
                ([lst[i] for i in indices], [lst[j] for j in range(n) if j not in indices])
                for k in range(1, n//2 + 1)
                for indices in combinations(range(n), k)
            ]
        else:
            all_combinations = [
                ([lst[i] for i in indices], [lst[j] for j in range(n) if j not in indices])
                for k in range(1, n // 2)
                for indices in combinations(range(n), k)
            ]
            all_combinations_aux = [
                ([lst[i] for i in indices], [lst[j] for j in range(n) if j not in indices])
                for pos, indices in enumerate(combinations(range(n), n // 2)) if pos <= math.factorial(n)/2 - 1
            ]
            all_combinations = all_combinations + all_combinations_aux

        return all_combinations

    def _calculate_impurity(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / np.sum(counts)
        impurity = 1 - np.sum(probabilities ** 2)
        return impurity

    def _most_common_class(self, y):
        classes, counts = np.unique(y, return_counts=True)
        return classes[np.argmax(counts)]

    def predict(self, X):
        return np.array([self._predict_single(x, self.tree) for x in X])

    def _predict_single(self, x, tree):
        if 'class' in tree:
            return tree['class']
        else:
            if x[tree['feature_index']] in tree['partition'][0]:
                return self._predict_single(x, tree['left'])
            else:
                return self._predict_single(x, tree['right'])
