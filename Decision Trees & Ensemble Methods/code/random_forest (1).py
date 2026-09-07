class RandomForest:
    def __init__(self, n_estimators=10, max_depth=10, min_samples_split=2, max_features='sqrt', criterion='entropy'):
        # Initialize ensemble parameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.trees = []

    def _bootstrap_sample(self, X, y):
        # Create a random sample with replacement (Bagging)
        n_samples = X.shape[0]
        # Randomly select row indices allowing duplicates
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def fit(self, X, y):
        # Train the ensemble by creating multiple independent trees
        self.trees = []
        for _ in range(self.n_estimators):
            # Create a new Decision Tree instance
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion
            )
            
            # Generate a unique bootstrap sample for this specific tree
            X_sample, y_sample = self._bootstrap_sample(X, y)
            
            # Train the individual tree
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        all_importances = np.array([tree.feature_importances_ for tree in self.trees])
        self.feature_importances_ = np.mean(all_importances, axis=0)

    def predict(self, X):
        # Collect predictions from every tree in the forest
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        
        # Reshape array to group predictions per sample (n_samples, n_trees)
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        
        # Use majority voting to pick the final class label
        y_pred = [self._most_common_label(sample_preds) for sample_preds in tree_preds]
        return np.array(y_pred)

    def _most_common_label(self, y):
        # Return the class with the highest frequency (Majority Vote)
        return np.bincount(y).argmax()


# 1. Load the Iris data
data = load_iris()
X, y = data.data, data.target

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize and train the Random Forest
rf = RandomForest(n_estimators=5, max_depth=10)
rf.fit(X_train, y_train)

# 4. Generate predictions and calculate accuracy score
predictions = rf.predict(X_test)
accuracy = np.mean(predictions == y_test)

print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")