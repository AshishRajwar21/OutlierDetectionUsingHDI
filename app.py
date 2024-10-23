
import math
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn import metrics

def find_local_density(distances, k, dimension):
    density_sum = 0
    for dist in distances:
        density_sum += ((1 / (2 * math.pi)) ** dimension) * math.exp(-0.5 * (dist) ** 2)
    return density_sum / k

def outlier_detection(X, k, m, dimension):
    # Step 1: Calculate k-NN graph
    knn_graph = NearestNeighbors(n_neighbors=k).fit(X)

    # Step 2: Initialize outlier scores
    outlier_scores = np.zeros(len(X))

    # Step 3: Calculate local density and outlier scores
    for i, x in enumerate(X):
        # Step 3.1: Obtain k-NN distances and indices
        distances, neighbors_indices = knn_graph.kneighbors([x], return_distance=True)

        # Step 3.2: Calculate local density
        p_x = find_local_density(distances[0], k, dimension)
        curr_max_density = p_x

        # Step 3.3: Iteratively update density and calculate outlier score
        for t in range(m):
            max_neighbor_density = float('-inf')
            max_density_index = 0
            for j, index in enumerate(neighbors_indices[0]):
                curr_distances, _ = knn_graph.kneighbors([X[index]], return_distance=True)
                curr_density = find_local_density(curr_distances[0], k, dimension)
                if curr_density > max_neighbor_density:
                    max_neighbor_density = curr_density
                    max_density_index = index

            if curr_max_density < max_neighbor_density:
                _, neighbors_indices = knn_graph.kneighbors([X[max_density_index]], return_distance=True)
                curr_max_density = max_neighbor_density
            else:
                break
        # Calculating COF or outlier scores
        outlier_scores[i] = curr_max_density / p_x

    return outlier_scores

# Read the CSV file into a DataFrame
df = pd.read_csv('DS04.txt')
# df = pd.read_csv('DS05_new.csv')
# df = pd.read_csv('DS02_new.csv')
X = df.values
new_X = X[:,0:2]

# Set parameters
k = 10
m = 10
dimension = X.shape[1] - 1

# Detect outliers
outlier_scores = outlier_detection(new_X, k, m, dimension)

# Identify threshold based on outlier_scores
a = 2.5
b = 1.4826

median_of_OS = np.median(outlier_scores)
OS_minus_median = np.abs(outlier_scores - median_of_OS)
median_of_OS_minus_median = np.median(OS_minus_median)

mad = b * median_of_OS_minus_median
threshold = median_of_OS + a * mad

# Identify outliers based on the threshold
outliers = np.where(outlier_scores > threshold)[0]
print(len(outliers))
# Generate the 'out' array for outliers
S = len(X)
out = np.zeros(S)
for i in range(S):
    if i in outliers:
        out[i] = 1

# Assuming the actual labels (Y) are in the third column of the dataset (adjust accordingly)
Y = X[:, 2]  # Replace this with the correct index if needed

# Binarize the labels for multiclass classification using One-vs-Rest (OvR)
n_classes = len(np.unique(Y))  # Count the number of unique classes
Y_bin = label_binarize(Y, classes=np.unique(Y))  # Convert labels to binary

# Plot the data (non-outliers and outliers)
plt.figure(figsize=(10, 6))

# Plot non-outliers
plt.scatter(X[~np.isin(np.arange(X.shape[0]), outliers), 0], X[~np.isin(np.arange(X.shape[0]), outliers), 1], 
            c='blue', label='Non-outliers', alpha=0.6)

# Plot outliers
plt.scatter(X[outliers, 0], X[outliers, 1], c='red', label='Outliers', alpha=0.6)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Outlier Detection')
plt.legend()
plt.grid(True)
plt.show()

# Compute ROC curve and AUC for each class
for i in range(n_classes):
    # Calculate ROC curve for the ith class
    fpr, tpr, _ = metrics.roc_curve(Y_bin[:, i], out)
    auc = metrics.roc_auc_score(Y_bin[:, i], out)
    
    # Plot ROC curve
    plt.plot(fpr, tpr, label=f"Class {i} AUC={auc:.2f}")

# Create plot for ROC curves of all classes
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curves for Multiclass')
plt.legend(loc=4)
plt.savefig('roc_curve_multiclass.png', dpi=2400)  # Save with high DPI
plt.show()

# Optionally, print AUC for each class
for i in range(n_classes):
    auc = metrics.roc_auc_score(Y_bin[:, i], out)
    print(f"AUC for class {i}: {auc}")




