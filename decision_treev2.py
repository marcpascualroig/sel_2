import numpy as np
from itertools import combinations
import math
import random
class DecisionTreeClassifier:
    def __init__(self, num_features, num_random_features=None, selected_feature_indices=None, max_depth=None, min_samples_split=1, min_impurity=-1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.num_features = num_features
        if num_random_features is None or num_random_features<0:
            self.num_random_features = 0
        else:
            self.num_random_features = num_random_features
        self.tree = None
        self.feature_frequencies = np.zeros(num_features)
        self.feature_frequencies_2 = np.zeros(num_features)

        if selected_feature_indices is not None:
            self.selected_feature_indices = selected_feature_indices
        else:
            self.selected_feature_indices = range(num_features)
        self.numerical_feature = np.full(num_features, False, dtype=bool)



    def fit(self, X, y):
        #generate tree
        for feature_index in range(self.num_features):
            if self._is_numeric_array(X[:, feature_index]):
                self.numerical_feature[feature_index] = True
        self.tree = self._build_tree(X, y)
        return self.feature_frequencies, self.feature_frequencies_2

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        unique_classes = np.unique(y)

        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or num_samples < self.min_samples_split or len(unique_classes) == 1:
            return {'class': self._most_common_class(y)}

        # Find best split (impurity)
        best_split, current_impurity = self._find_best_split(X, y)
        #print(best_split)

        #stopping criteria, impurity threshold
        if len(best_split)==0 or best_split['impurity'] <= self.min_impurity:
            return {'class': self._most_common_class(y)}

        #add counting for feature importance
        self.feature_frequencies[self.selected_feature_indices.index(best_split['feature_index'][0])] += 1
        self.feature_frequencies_2[self.selected_feature_indices.index(best_split['feature_index'][0])] += num_samples*(current_impurity - best_split['impurity'])

        left_subtree = self._build_tree(*best_split['left'], depth + 1)
        right_subtree = self._build_tree(*best_split['right'], depth + 1)

        return {'feature_index': best_split['feature_index'],
                'partition': best_split['values_subsets'],
                'left': left_subtree,
                'right': right_subtree}

    def _find_best_split(self, X, y):
        num_samples, num_features = X.shape
        best_split = {}
        best_impurity = self._calculate_impurity(y)
        current_imputiry = best_impurity

        #random selection of features
        if self.num_random_features >0:
            features = random.sample(range(num_features), self.num_random_features)
        else:
            features = range(self.num_features)
        partitions = [None] * len(features)
        for i, feature_index in enumerate(features):
            if self.numerical_feature[feature_index]:
                feature_values = np.unique(X[:, feature_index])
                partitions[i] = self._get_thresholds_numerical(feature_values)
            else:
                feature_values = np.unique(X[:, feature_index].astype(str))
                partitions[i] = self._get_combinations_categorical(feature_values)

        #iterate for all features
        for i, feature_index in enumerate(features):
            for partition in partitions[i]:
                if self.numerical_feature[feature_index]:
                    threshold = partition
                    left_indices = np.where(X[:, feature_index] <= threshold)
                    right_indices = np.where(X[:, feature_index] > threshold)

                else:
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
                        'feature_index': [self.selected_feature_indices[feature_index], self.numerical_feature[feature_index]],
                        'values_subsets': partition,
                        'left': (X[left_indices], y[left_indices]),
                        'right': (X[right_indices], y[right_indices]),
                        'impurity': impurity
                    }
                    best_impurity = impurity
        return best_split, current_imputiry


    def _get_combinations_categorical(self, lst):
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

    def _get_thresholds_numerical(self, lst):
        midpoints = []
        for i in range(len(lst) - 1):
            # Calculate the midpoint between consecutive unique values
            midpoint = (lst[i] + lst[i + 1]) / 2.0
            midpoints.append(midpoint)
        return midpoints


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
            #numerical
            if tree['feature_index'][1]:
                if x[tree['feature_index'][0]] < tree['partition']:
                    return self._predict_single(x, tree['left'])
                else:
                    return self._predict_single(x, tree['right'])
            #categorical
            else:
                if x[tree['feature_index'][0]] in tree['partition'][0]:
                    return self._predict_single(x, tree['left'])
                elif x[tree['feature_index'][0]] in tree['partition'][1]:
                    return self._predict_single(x, tree['right'])
                elif len(tree['left'])>len(tree['right']):
                    return self._predict_single(x, tree['left'])
                else:
                    return self._predict_single(x, tree['right'])

    def _is_numeric_array(self, arr):
        try:
            arr = arr.astype(float)
            return True
        except ValueError:
            return False
