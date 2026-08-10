"""Shared tree construction and traversal."""

from dataclasses import dataclass

import numpy as np


@dataclass
class TreeNode:
    """A decision node or terminal leaf in a fitted tree."""

    node_id: int
    depth: int
    n_samples: int
    impurity: float
    value: object
    distribution: object = None
    feature_index: object = None
    threshold: object = None
    left: object = None
    right: object = None

    @property
    def is_leaf(self):
        return self.left is None and self.right is None


class BaseDecisionTree:
    """Greedy, binary CART-style tree shared by classification and regression."""

    def __init__(
        self,
        *,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        min_impurity_decrease=0.0,
        random_state=None,
    ):
        if max_depth is not None and (
            not isinstance(max_depth, int) or max_depth <= 0
        ):
            raise ValueError("max_depth must be None or a positive integer")
        if not isinstance(min_samples_split, int) or min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer of at least 2")
        if not isinstance(min_samples_leaf, int) or min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be a positive integer")
        if min_impurity_decrease < 0:
            raise ValueError("min_impurity_decrease must be non-negative")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state

        self.tree_ = None
        self.n_features_in_ = None
        self.n_samples_fit_ = None
        self.max_features_ = None
        self.feature_importances_ = None
        self._raw_feature_importances = None
        self._rng = None
        self._next_node_id = 0

    @staticmethod
    def _validate_X(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a non-empty 2D array")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        return X

    @classmethod
    def _validate_fit_data(cls, X, y, *, numeric_target):
        X = cls._validate_X(X)
        y = np.asarray(y, dtype=float if numeric_target else None)
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples")
        if numeric_target and not np.all(np.isfinite(y)):
            raise ValueError("y must contain only finite values")
        if not numeric_target and np.issubdtype(y.dtype, np.number):
            if not np.all(np.isfinite(y)):
                raise ValueError("y must contain only finite labels")
        return X, y

    def _resolve_max_features(self):
        value = self.max_features
        if value is None:
            return self.n_features_in_
        if value == "sqrt":
            return max(1, int(np.sqrt(self.n_features_in_)))
        if value == "log2":
            return max(1, int(np.log2(self.n_features_in_)))
        if isinstance(value, int) and not isinstance(value, bool):
            if not 1 <= value <= self.n_features_in_:
                raise ValueError("integer max_features must be between 1 and n_features")
            return value
        if isinstance(value, float):
            if not 0 < value <= 1:
                raise ValueError("float max_features must be in (0, 1]")
            return max(1, int(value * self.n_features_in_))
        raise ValueError("max_features must be None, int, float, 'sqrt', or 'log2'")

    def _fit_tree(self, X, y):
        self.n_features_in_ = X.shape[1]
        self.n_samples_fit_ = X.shape[0]
        self.max_features_ = self._resolve_max_features()
        self._raw_feature_importances = np.zeros(self.n_features_in_, dtype=float)
        self._rng = np.random.default_rng(self.random_state)
        self._next_node_id = 0
        self.tree_ = self._build_tree(X, y, depth=0)
        total = np.sum(self._raw_feature_importances)
        self.feature_importances_ = (
            self._raw_feature_importances / total
            if total > 0
            else self._raw_feature_importances.copy()
        )
        return self

    def _new_node(self, *, depth, y, impurity):
        node = TreeNode(
            node_id=self._next_node_id,
            depth=depth,
            n_samples=len(y),
            impurity=float(impurity),
            value=self._leaf_value(y),
            distribution=self._leaf_distribution(y),
        )
        self._next_node_id += 1
        return node

    def _build_tree(self, X, y, depth):
        impurity = self._impurity(y)
        node = self._new_node(depth=depth, y=y, impurity=impurity)
        reached_depth = self.max_depth is not None and depth >= self.max_depth
        if (
            reached_depth
            or len(y) < self.min_samples_split
            or len(y) < 2 * self.min_samples_leaf
            or impurity <= np.finfo(float).eps
        ):
            return node

        feature, threshold, gain = self._best_split(X, y, impurity)
        weighted_gain = (len(y) / self.n_samples_fit_) * gain
        if feature is None or weighted_gain + 1e-15 < self.min_impurity_decrease:
            return node

        left_mask = X[:, feature] <= threshold
        node.feature_index = feature
        node.threshold = float(threshold)
        self._raw_feature_importances[feature] += len(y) * gain
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def _best_split(self, X, y, parent_impurity):
        if self.max_features_ == self.n_features_in_:
            features = np.arange(self.n_features_in_)
        else:
            features = np.sort(
                self._rng.choice(
                    self.n_features_in_, size=self.max_features_, replace=False
                )
            )

        best_feature = None
        best_threshold = None
        best_gain = 0.0
        for feature in features:
            threshold, gain = self._best_feature_split(
                X[:, feature], y, parent_impurity
            )
            if gain > best_gain + 1e-15:
                best_feature = int(feature)
                best_threshold = threshold
                best_gain = gain
        return best_feature, best_threshold, best_gain

    def _check_is_fitted(self):
        if self.tree_ is None:
            raise ValueError("No model yet. Call fit() first.")

    def _validate_query(self, X):
        self._check_is_fitted()
        X = self._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the training data")
        return X

    def _leaf_for_sample(self, sample):
        node = self.tree_
        while not node.is_leaf:
            node = node.left if sample[node.feature_index] <= node.threshold else node.right
        return node

    def apply(self, X):
        """Return the terminal node identifier reached by every sample."""
        X = self._validate_query(X)
        return np.asarray([self._leaf_for_sample(row).node_id for row in X], dtype=int)

    def get_depth(self):
        """Return the maximum depth of the fitted tree."""
        self._check_is_fitted()
        return self._max_depth(self.tree_)

    def _max_depth(self, node):
        if node.is_leaf:
            return node.depth
        return max(self._max_depth(node.left), self._max_depth(node.right))

    def get_n_leaves(self):
        """Return the number of terminal nodes."""
        self._check_is_fitted()
        return self._count_leaves(self.tree_)

    def _count_leaves(self, node):
        if node.is_leaf:
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)

    def export_text(self, feature_names=None, decimals=3):
        """Return a compact, human-readable representation of the tree."""
        self._check_is_fitted()
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(self.n_features_in_)]
        if len(feature_names) != self.n_features_in_:
            raise ValueError("feature_names must match the number of fitted features")
        lines = []

        def visit(node, prefix):
            if node.is_leaf:
                lines.append(f"{prefix}predict: {self._format_value(node.value)}")
                return
            name = feature_names[node.feature_index]
            threshold = round(node.threshold, decimals)
            lines.append(f"{prefix}if {name} <= {threshold}:")
            visit(node.left, prefix + "  ")
            lines.append(f"{prefix}else:")
            visit(node.right, prefix + "  ")

        visit(self.tree_, "")
        return "\n".join(lines)

    def _format_value(self, value):
        return str(value)

    def _impurity(self, y):
        raise NotImplementedError

    def _best_feature_split(self, values, y, parent_impurity):
        raise NotImplementedError

    def _leaf_value(self, y):
        raise NotImplementedError

    def _leaf_distribution(self, y):
        return None
