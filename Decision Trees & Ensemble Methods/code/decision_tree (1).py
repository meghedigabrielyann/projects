def entropy(y):
    # Calculate impurity using Shannon Entropy
    proportions = np.bincount(y) / len(y)
    return -np.sum([p * np.log2(p) for p in proportions if p > 0])

def gini(y):
    # Calculate impurity using Gini index
    proportions = np.bincount(y) / len(y)
    return 1 - np.sum([p**2 for p in proportions])

class Node:
    # Container for tree structure (decision split or leaf result)
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature      # Index of the splitting column
        self.threshold = threshold  # Numeric threshold for the split
        self.left = left            # Left branch (<= threshold)
        self.right = right          # Right branch (> threshold)
        self.value = value          # Class label (only for leaf nodes)

    def is_leaf_node(self):
        # Check if node is a terminal leaf
        return self.value is not None

class DecisionTree:
    # Decision Tree Classifier built from scratch
    def __init__(self, min_samples_split=2, max_depth=100, criterion='entropy'):
        self.min_samples_split = min_samples_split # Min samples to justify a split
        self.max_depth = max_depth                 # Limit tree depth
        self.criterion = criterion                 # 'entropy' or 'gini'
        self.root = None                           # Root of the tree


    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.root = self._grow_tree(X, y)
        self.feature_importances_ = np.zeros(self.n_features_)
        self._calculate_importances(self.root)

    def _calculate_importances(self, node):
        if node.value is not None: return
        self.feature_importances_[node.feature] += 1
        self._calculate_importances(node.left)
        self._calculate_importances(node.right)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # 1. Stop if max depth, pure node, or not enough samples
        if (self.max_depth is not None and depth >= self.max_depth) or n_labels == 1 or n_samples < self.min_samples_split:
            return Node(value=self._most_common_label(y))

        # 2. Pick features and find the best split point
        feat_idxs = np.random.choice(n_feats, n_feats, replace=False)
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)

        # 3. Create branches and grow subtrees recursively
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        
        return Node(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        # Greedy search for split with highest Information Gain
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thresh in thresholds:
                gain = self._information_gain(y, X_column, thresh)
                if gain > best_gain:
                    best_gain, split_idx, split_threshold = gain, feat_idx, thresh

        return split_idx, split_threshold
        self.feature_importances_[best_feature_idx] += (n_samples / total_samples) * best_gain

    def _information_gain(self, y, X_column, threshold):
        # Reduction in impurity: Parent loss - Weighted Children loss
        parent_loss = self._entropy(y) if self.criterion == 'entropy' else self._gini(y)
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        n, n_l, n_r = len(y), len(left_idxs), len(right_idxs)
        e_l = self._entropy(y[left_idxs]) if self.criterion == 'entropy' else self._gini(y[left_idxs])
        e_r = self._entropy(y[right_idxs]) if self.criterion == 'entropy' else self._gini(y[right_idxs])
        
        child_loss = (n_l / n) * e_l + (n_r / n) * e_r
        return parent_loss - child_loss

    def _entropy(self, y):
        # Helper for Shannon Entropy
        proportions = np.bincount(y) / len(y)
        return -np.sum([p * np.log2(p) for p in proportions if p > 0]) 

    def _gini(self, y):
        # Helper for Gini Impurity
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum([p**2 for p in proportions])

    def _split(self, X_column, split_thresh):
        # Separate data indices based on threshold
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs
  
    def predict(self, X):
        # Predict labels for a set of inputs
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        # Move down branches until a leaf is reached
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _most_common_label(self, y):
        if len(y) == 0:
            return 0 
        # Majority voting for the final prediction at leaf
        return np.bincount(y).argmax()

# 1. Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# 2. Split data into Train and Test sets (80/20 ratio)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize your DecisionTree model
# Max depth 10 is usually enough for Iris
tree = DecisionTree(max_depth=10, criterion='entropy') 

# 4. Train the model on training data
tree.fit(X_train, y_train)

# 5. Generate predictions for the test set
predictions = tree.predict(X_test)

# 6. Calculate accuracy
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy * 100} %")

# Verify if accuracy meets the > 95% requirement
if accuracy > 0.95:
    print("Congratulations! Your model meets the requirements.")
else:
    print("Accuracy is below 95%. Check your split logic.")