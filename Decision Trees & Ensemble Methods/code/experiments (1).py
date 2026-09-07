# Define dataset columns
columns = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", 
           "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss", 
           "hours_per_week", "native_country", "income"]

# Load dataset and clean missing values
df = pd.read_csv("adult.data", names=columns, skipinitialspace=True)
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

# Convert target labels to binary (1 for >50K, 0 for <=50K)
df['income'] = df['income'].apply(lambda x: 1 if '>50K' in x else 0)

# Factorize categorical strings into numeric values
for col in df.select_dtypes(include=['object']).columns:
    df[col] = pd.factorize(df[col])[0]

# ---  DATA PREPARATION ---
# Using 3000 samples as per your setup for speed
df_sample = df.sample(3000, random_state=42)
X = df_sample.drop('income', axis=1).values
y = df_sample['income'].values

# Required 80/20 split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- EXPERIMENT 1 ---
# Defining the 4 required models 
models = {
    "My DT": DecisionTree(max_depth=10),
    "My RF": RandomForest(n_estimators=20, max_depth=10),
    "Sklearn DT": SklearnDT(max_depth=10),
    "Sklearn RF": SklearnRF(n_estimators=20, max_depth=10)
}

results = []

print(f"{'Model':<15} | {'Train Acc':<10} | {'Test Acc':<10} | {'Train Time':<10} | {'Pred Time':<10}")
print("-" * 70)

for name, model in models.items():
    # Measure Training Time 
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train
    
    # Measure Prediction Time 
    start_pred = time.time()
    y_test_pred = model.predict(X_test)
    pred_time = time.time() - start_pred
    
    # Calculate Accuracies (Training & Test) 
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    # Document Number of Parameters 
    params = "depth=10" if "DT" in name else "trees=20, depth=10"
    
    results.append({
        "Model": name,
        "Train Acc": train_acc,
        "Test Acc": test_acc,
        "Train Time": train_time,
        "Pred Time": pred_time,
        "Params": params
    })
    
    print(f"{name:<15} | {train_acc:<10.4f} | {test_acc:<10.4f} | {train_time:<10.4f}s | {pred_time:.4f}s")

# Convert results to DataFrame for analysis [cite: 226]
results_df = pd.DataFrame(results)

# --- Model Performance Comparison ---
plt.figure(figsize=(10, 5))

# Create a bar chart comparing Custom and Sklearn implementations
# Use different colors to distinguish between DT and RF models
plt.bar(results_df['Model'], results_df['Test Acc'], 
        color=['salmon', 'darkred', 'lightblue', 'darkblue'])

plt.title('Test Accuracy Comparison: Custom vs Sklearn')
plt.ylabel('Accuracy Score')

# Set y-axis limits to zoom in on the performance differences
plt.ylim(0.7, 1.0) 

# Add a horizontal grid for better readability of accuracy values
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# --- Overfitting Analysis: Train vs. Test Accuracy ---
# Set the position and width for grouped bars
x = np.arange(len(results_df['Model']))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))

# Plot training accuracy bars (shorter/lighter color for context)
ax.bar(x - width/2, results_df['Train Acc'], width, 
       label='Train Accuracy', color='gray', alpha=0.6)

# Plot testing accuracy bars (bolder color for comparison)
ax.bar(x + width/2, results_df['Test Acc'], width, 
       label='Test Accuracy', color='blue', alpha=0.8)

# Add descriptive labels and title
ax.set_ylabel('Accuracy')
ax.set_title('Evidence of Overfitting: Train vs Test Accuracy')

# Set model names on the x-axis
ax.set_xticks(x)
ax.set_xticklabels(results_df['Model'])

# Add a legend to differentiate the sets
ax.legend()
plt.show()

# --- Training Efficiency Analysis ---
plt.figure(figsize=(10, 5))

# Plot training time for each model using a bar chart
plt.bar(results_df['Model'], results_df['Train Time'], color='purple')

plt.title('Training Time Comparison (Seconds)')
plt.ylabel('Seconds')

# Use log scale to handle large differences between Custom and Sklearn speeds
plt.yscale('log') 
plt.show()

# --- EXPERIMENT 2 ---

#DECISION TREE TUNING

depths = [1, 2, 3, 5, 10, 15, 20, None]
min_samples = [2, 5, 10, 20, 50]
dt_heatmap_data = np.zeros((len(depths), len(min_samples)))

for i, d in enumerate(depths):
    for j, ms in enumerate(min_samples):
        model = DecisionTree(max_depth=d, min_samples_split=ms)
        model.fit(X_train, y_train)
        dt_heatmap_data[i, j] = accuracy_score(y_test, model.predict(X_test))

# Criterion Comparison
criteria = ['gini', 'entropy']
crit_scores = []
for crit in criteria:
    model = DecisionTree(criterion=crit, max_depth=10)
    model.fit(X_train, y_train)
    crit_scores.append(accuracy_score(y_test, model.predict(X_test)))

#RANDOM FOREST TUNING
n_trees = [1, 5, 10, 25, 50, 100, 200]
rf_train_acc, rf_test_acc = [], []
for n in n_trees:
    model = RandomForest(n_estimators=n, max_depth=10)
    model.fit(X_train, y_train)
    rf_train_acc.append(accuracy_score(y_train, model.predict(X_train)))
    rf_test_acc.append(accuracy_score(y_test, model.predict(X_test)))

# Max Features Impact
max_feat_options = [1, 'sqrt', 'log2', X_train.shape[1]] 
feat_results = []
for feat in max_feat_options:
    model = RandomForest(n_estimators=50, max_features=feat, max_depth=10)
    model.fit(X_train, y_train)
    feat_results.append(accuracy_score(y_test, model.predict(X_test)))

print("\n" + "="*30)
print("--- EXPERIMENT 2 RESULTS ---")
for i, crit in enumerate(criteria):
    print(f"Criterion: {crit:<8} | Accuracy: {crit_scores[i]:.4f}")

best_idx = np.unravel_index(np.argmax(dt_heatmap_data, axis=None), dt_heatmap_data.shape)
print(f"\nBest DT Params: depth={depths[best_idx[0]]}, min_samples={min_samples[best_idx[1]]}")
print(f"Best DT Accuracy: {dt_heatmap_data[best_idx]:.4f}")

best_n_idx = np.argmax(rf_test_acc)
print(f"Best RF n_estimators: {n_trees[best_n_idx]} | Accuracy: {rf_test_acc[best_n_idx]:.4f}")

# --- 1. GRID SEARCH HEATMAP ---
# Visualizes the combined impact of depth and split size on accuracy.
plt.figure(figsize=(10, 8), dpi=300)
sns.heatmap(dt_heatmap_data, annot=True, fmt=".4f", 
            xticklabels=min_samples, 
            yticklabels=[str(d) for d in depths], cmap='viridis')
plt.title('Figure 4: Grid Search Heatmap (Depth vs Min Samples Split)', fontsize=14)
plt.xlabel('min_samples_split', fontsize=12)
plt.ylabel('max_depth', fontsize=12)
plt.show()

# --- 2. LEARNING CURVES ---
# Shows how Random Forest performance stabilizes as more trees are added.
plt.figure(figsize=(10, 6), dpi=300)
plt.plot(n_trees, rf_train_acc, 'o-', color="r", label="Training Accuracy")
plt.plot(n_trees, rf_test_acc, 's-', color="g", label="Testing Accuracy")
plt.title('Figure 5: Learning Curves (Random Forest n_estimators)', fontsize=14)
plt.xlabel('Number of Trees (n_estimators)', fontsize=12)
plt.ylabel('Accuracy Score', fontsize=12)
plt.legend(loc="best")
plt.grid(True, alpha=0.3)
plt.show()

# --- 3. VALIDATION CURVES ---
# Analyzes how different feature selection strategies affect model accuracy.
plt.figure(figsize=(10, 6), dpi=300)
feat_labels = ['1', 'sqrt', 'log2', 'all']
plt.plot(feat_labels, feat_results, 'D-', color="blue", linewidth=2)
plt.title('Figure 6: Validation Curve (Max Features Impact)', fontsize=14)
plt.xlabel('Max Features Strategy', fontsize=12)
plt.ylabel('Test Accuracy', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# --- EXPERIMENT 3 ---

# --- 1. FEATURE IMPORTANCE ---
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# --- 2. TOP FEATURES PLOT ---
top_n = min(10, len(importances))
plt.figure()
plt.bar(range(top_n), importances[indices[:top_n]])
plt.xticks(range(top_n), indices[:top_n])
plt.title("Top Feature Importances")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.show()

# --- 3. TRAIN WITH TOP-K FEATURES ---
k_values = [1, 3, 5, 10]
subset_results = []

# Baseline (all features)
full_model = RandomForestClassifier(n_estimators=50, random_state=42)
full_model.fit(X_train, y_train)
y_pred_full = full_model.predict(X_test)
full_acc = accuracy_score(y_test, y_pred_full)

print("Full model accuracy:", full_acc)

for k in k_values:
    selected_features = indices[:k]

    X_train_k = X_train[:, selected_features]
    X_test_k = X_test[:, selected_features]

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train_k, y_train)

    y_pred = model.predict(X_test_k)
    acc = accuracy_score(y_test, y_pred)

    subset_results.append(acc)

    print(f"k = {k}, Accuracy = {acc}")

# --- 4. PERFORMANCE PLOT ---
plt.figure(figsize=(8,5))
plt.plot(k_values, subset_results, marker='o', label='Top-k features')
plt.axhline(y=full_acc, linestyle='--', label='All features')

plt.xlabel("Number of Features (k)")
plt.ylabel("Accuracy")
plt.title("Performance vs Number of Features")
plt.legend()
plt.grid()
plt.show()


print("\nTop Feature Importances:")
for i in range(top_n):
    idx = indices[i]
    print(f"{i+1}. Feature {idx} → Importance = {importances[idx]:.4f}")

print("\n--- Model Performance ---")

# Full model
print(f"All features accuracy: {full_acc:.4f}")

# Top-k results
for i in range(len(k_values)):
    print(f"Top-{k_values[i]} features accuracy: {subset_results[i]:.4f}")

print("\n--- Comparison Summary ---")

for i in range(len(k_values)):
    diff = full_acc - subset_results[i]
    print(f"k = {k_values[i]} → Accuracy = {subset_results[i]:.4f} | Drop = {diff:.4f}")

# --- Bias-Variance Analysis ---
def plot_learning_curve(model, X_train, y_train, X_test, y_test, title):
    # Define training sizes from 10% to 100%
    train_sizes = np.linspace(0.1, 1.0, 5)
    train_accs, test_accs = [], []
    
    for size in train_sizes:
        # Select a subset of the training data
        num_samples = int(size * len(X_train))
        X_subset, y_subset = X_train[:num_samples], y_train[:num_samples]
        
        # Train model and record accuracy for both sets
        model.fit(X_subset, y_subset)
        train_accs.append(accuracy_score(y_subset, model.predict(X_subset)))
        test_accs.append(accuracy_score(y_test, model.predict(X_test)))
        
    # Plot performance curves
    plt.plot(train_sizes * 100, train_accs, 'o-', label="Train Accuracy")
    plt.plot(train_sizes * 100, test_accs, 's-', label="Test Accuracy")
    plt.title(title)
    plt.xlabel("Training Set Size (%)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

# Comparison: Single Tree vs. Random Forest
plt.figure(figsize=(12, 5))

# Subplot 1: Shows high variance (large gap between train and test)
plt.subplot(1, 2, 1)
plot_learning_curve(DecisionTree(max_depth=20), X_train, y_train, X_test, y_test, "Single Tree (High Variance)")

# Subplot 2: Shows reduced variance (smaller gap, better generalization)
plt.subplot(1, 2, 2)
plot_learning_curve(RandomForest(n_estimators=20, max_depth=10), X_train, y_train, X_test, y_test, "Random Forest (Reduced Variance)")

plt.tight_layout()
plt.show()

# --- Computational Complexity ---
# --- 1. Decision Tree: Training Time vs. Depth ---
depths = [3, 5, 10, 15, 20]
train_times = []

for d in depths:
    start = time.time()
    dt = DecisionTree(max_depth=d)
    dt.fit(X_train, y_train)
    # Measure time taken to grow the tree to depth d
    train_times.append(time.time() - start)

plt.figure(figsize=(8, 5))
plt.plot(depths, train_times, 'go-')
plt.title("Training Time vs. Tree Depth")
plt.xlabel("Max Depth")
plt.ylabel("Time (seconds)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# --- 2. Random Forest: Training Time vs. Number of Trees ---
n_trees = [1, 5, 10, 20, 50]
rf_times = []

for n in n_trees:
    start = time.time()
    # Measure total ensemble training time for n estimators
    rf_temp = RandomForest(n_estimators=n, max_depth=5)
    rf_temp.fit(X_train, y_train)
    rf_times.append(time.time() - start)

plt.figure(figsize=(8, 5))
plt.plot(n_trees, rf_times, 'ro-')
plt.title("Training Time vs. Number of Trees (RF)")
plt.xlabel("n_estimators")
plt.ylabel("Time (seconds)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


# --- Decision Boundary Visualisation ---
# --- 1. Dimensionality Reduction ---
# Use PCA to reduce features to 2D for visualization purposes
pca = PCA(n_components=2)
# Using a subset of 500 samples for faster plotting
X_train_pca = pca.fit_transform(X_train[:500]) 
y_subset = y_train[:500]

def plot_boundaries(model, X, y, title):
    # Fit the model on the 2D PCA-transformed data
    model.fit(X, y)
    
    # Define the mesh grid for the plot background
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    
    # Predict values across the entire mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision regions and actual data points
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
    plt.title(title)

# --- 2. Plotting Comparison ---
plt.figure(figsize=(12, 5))

# Single Tree: Usually shows sharp, step-like boundaries
plt.subplot(1, 2, 1)
plot_boundaries(DecisionTree(max_depth=5), X_train_pca, y_subset, "DT Decision Boundary")

# Random Forest: Usually shows smoother, averaged boundaries
plt.subplot(1, 2, 2)
plot_boundaries(RandomForest(n_estimators=10, max_depth=5), X_train_pca, y_subset, "RF Decision Boundary")

plt.show()