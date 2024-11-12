

# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 15:23:09 2024

@author: yrakh
"""

import math
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import os

def find_local_density(distances, k, dimension):
    density_sum = 0
    for dist in distances:
        density_sum += ((1 / (2 * math.pi)) ** dimension) * math.exp(-0.5 * ((dist) ** 2))
    return density_sum / k


def outlier_detection(X, k, dimension, density_percentile=10):
    # Step 1: Calculate k-NN graph
    knn_graph = NearestNeighbors(n_neighbors=k+1).fit(X)
    outlier_scores = np.zeros(len(X))
    alpha = 0.05
    local_densities = []
    temp_local_densities = []
    distance_matrix = []
    neighbor_index_matrix = []

    for i, x in enumerate(X):
        distances, neighbors_indices = knn_graph.kneighbors([x], return_distance=True)
        distance_matrix.append(distances[0])
        neighbor_index_matrix.append(neighbors_indices[0])
        p_x = find_local_density(distances[0], k, dimension)
        local_densities.append(p_x)
        temp_local_densities.append((p_x, i))

    # Step 2: Sort densities and select the lowest density points (bottom 10%)
    temp_local_densities.sort(key=lambda x: x[0])
    num_low_density_points = int(len(X) * (density_percentile / 100.0))
    low_density_indices = [index for _, index in temp_local_densities[:num_low_density_points]]
    low_density_indices_set = set(low_density_indices)

    # Calculate high-density indices (remaining 90%)
    high_density_indices = [index for _, index in temp_local_densities[num_low_density_points:]]

    # Step 3: Calculate outlier scores for low-density points
    for i in low_density_indices:
        p_x = local_densities[i]
        curr_max_density = p_x
        neighbors_indices = neighbor_index_matrix[i]
        first_neighbor_density = p_x

        while True:
            max_neighbor_density = max(local_densities[index] for index in neighbors_indices)
            max_density_index = neighbors_indices[np.argmax([local_densities[index] for index in neighbors_indices])]

            if curr_max_density < max_neighbor_density:
                if first_neighbor_density == p_x:
                    first_neighbor_density = max_neighbor_density
                neighbors_indices = neighbor_index_matrix[max_density_index]
                curr_max_density = max_neighbor_density
                if max_density_index not in low_density_indices_set:
                    break
            else:
                break

        outlier_scores[i] = ((1 - alpha) * curr_max_density + alpha * p_x) / p_x

    return low_density_indices, high_density_indices, local_densities, outlier_scores

# Load data
def load_data(file_path):
    _, file_extension = os.path.splitext(file_path)
    if file_extension == '.csv':
        data = pd.read_csv(file_path).values
    elif file_extension == '.txt':
        data = np.loadtxt(file_path)
    else:
        raise ValueError("Unsupported file extension")
    return data

# Read the CSV file into a DataFrame
# file_to_load = r"D:\spyder\Data\2dim\DS01.txt"
# file_to_load = r"D:\spyder\Data\2dim\DS02.txt"
# file_to_load = r"D:\spyder\Data\2dim\DS03.txt"
# file_to_load = r"D:\spyder\Data\2dim\DS04.txt"
# file_to_load = r"D:\spyder\Data\2dim\DS05.txt"
# file_to_load = r"D:\spyder\Data\2dim\DS06.txt"
# file_to_load = r"D:\spyder\Data\2dim\DS07.txt"
# file_to_load = r"D:\spyder\Data\2dim\DS08.txt"
# file_to_load = r"D:\spyder\Data\2dim\DS09.txt"
# file_to_load = r"D:\spyder\Data\2dim\DS10.txt"
# file_to_load = r"D:\spyder\Data\2dim\DS11.txt"
# file_to_load = r"D:\spyder\Data\2dim\DS12.txt"
file_to_load = "DS02.txt"
file_to_load = "DS03.txt"


X = load_data(file_to_load)
new_X = X[:, :-1]  # Assuming all but the last column are features
actual_outliers = X[:, -1]  # Assuming the last column is the actual outlier label
k = 10
dimension = new_X.shape[1]

# Detect outliers using 10% least dense points
low_density_indices,high_density_indices, local_densities, outlier_scores = outlier_detection(new_X, k, dimension, density_percentile=10)

# Calculate threshold based on outlier scores
new_outlier_scores = []
# new_outlier_scores = [score for i, score in enumerate(outlier_scores) if i in low_density_indices]
for i, score in enumerate(outlier_scores) :
    if i in low_density_indices :
        new_outlier_scores.append(score)

median_of_OS = np.median(new_outlier_scores)
threshold = median_of_OS + np.mean((new_outlier_scores - median_of_OS) ** 2) / np.mean(new_outlier_scores)

outlier_class = []
# outlier_class = [i for i in range(len(outlier_scores)) if outlier_scores[i] > threshold]
for i in range(len(outlier_scores)): 
    if outlier_scores[i] > threshold :
        outlier_class.append(i)

# Step 1: Directly classify points as outliers if they are in both the low-density and actually outlier in dataset
refined_outliers = set()
for idx in outlier_class:
    for i in low_density_indices:
        if idx == i:
            if actual_outliers[i] == 1:
                refined_outliers.add(i)

# Create separate arrays for refined outliers and inliers
refined_outliers_list = list(refined_outliers)

# Remaining points to classify based on proximity to refined outliers/inliers
remaining_outliers = []
# remaining_outliers = [i for i in range(len(outlier_scores)) if i not in refined_outliers]
for i in range(len(outlier_scores)): 
    if i not in refined_outliers :
        remaining_outliers.append(i)


inliers_indices = high_density_indices

# Ensure that inliers_indices is not empty
# Initialize k-NN model for inliers if there are inliers
if inliers_indices:
    knn_inliers = NearestNeighbors(n_neighbors=1).fit(new_X[inliers_indices])
else:
    print("No inliers found. Skipping inliers k-NN fitting.")
    knn_inliers = None  # Set to None or handle as needed

# Initialize k-NN model for refined outliers
knn_outliers = NearestNeighbors(n_neighbors=1).fit(new_X[refined_outliers_list])

# Refine classification of remaining outliers based on the density of the nearest point
for i in remaining_outliers:
    # Find the nearest refined outlier's index and its density
    
    _, outlier_neighbor_index = knn_outliers.kneighbors([new_X[i]])
    print(outlier_neighbor_index)
    nearest_outlier_index = refined_outliers_list[outlier_neighbor_index[0][0]]
    # nearest_outlier_index = outlier_neighbor_index[0][0]
    nearest_outlier_density = local_densities[nearest_outlier_index]
    print(i,' ',nearest_outlier_index)

    # Calculate the density of the nearest inlier, if any inliers exist
    if knn_inliers is not None:
        _, inlier_neighbor_index = knn_inliers.kneighbors([new_X[i]])
        nearest_inlier_index = inliers_indices[inlier_neighbor_index[0][0]]
        # nearest_inlier_index = inlier_neighbor_index[0][0]
        nearest_inlier_density = local_densities[nearest_inlier_index]
        print(i,' ',nearest_inlier_index)
    else:
        nearest_inlier_density = -np.inf  # Set to a very low value if no inliers exist

    # Classify based on the density of the nearest point
    if nearest_outlier_density > nearest_inlier_density:
        refined_outliers.add(i)  # Classified as an outlier if closer in density to refined outliers


# Continue with plotting and further processing...


# Plot results
# Plot actual data points (Plot 1)
plt.figure(figsize=(10, 6))
plt.scatter(X[X[:, 2] == 0, 0], X[X[:, 2] == 0, 1], c='blue', label='Class 0', alpha=0.6)
plt.scatter(X[X[:, 2] == 1, 0], X[X[:, 2] == 1, 1], c='red', label='Class 1', alpha=0.6)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot based on Actual Outlier Data (Class 0 or 1)')
plt.legend()
plt.grid(True)
plt.show()

# Plot low and high-density points (Plot 2)
plt.figure(figsize=(10, 6))
plt.scatter(X[~np.isin(np.arange(X.shape[0]), low_density_indices), 0], 
            X[~np.isin(np.arange(X.shape[0]), low_density_indices), 1], 
            c='blue', label='High-density Points', alpha=0.6)
plt.scatter(X[low_density_indices, 0], X[low_density_indices, 1], c='red', label='Low-density Points', alpha=0.6)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot for Low and High-density Points')
plt.legend()
plt.grid(True)
plt.show()

# Plot points classified as outliers based on the threshold
plt.figure(figsize=(10, 6))

# Plot all points as blue by default (Non-outliers)
plt.scatter(X[:, 0], X[:, 1], c='blue', label='Non-outliers', alpha=0.6)

# Plot 10% low-density points in black
plt.scatter(X[low_density_indices, 0], X[low_density_indices, 1], 
            c='black', label='10% Low-density Points', alpha=0.6)

# Plot points classified as outliers based on the threshold in red
# plt.scatter(X[outlier_scores > threshold, 0], X[outlier_scores > threshold, 1], 
#             c='red', label='Threshold-based Outliers', alpha=0.6)

# Plot ignored points in orange
# plt.scatter(X[list(ignore_points), 0], X[list(ignore_points), 1], 
#             c='orange', label='Ignored Points', alpha=0.6)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Outliers Based on Threshold Value with Additional Categories')
plt.legend()
plt.grid(True)
plt.show()

# Plot points classified as outliers based on the threshold
plt.figure(figsize=(10, 6))

# Plot all points as blue by default (Non-outliers)
plt.scatter(X[:, 0], X[:, 1], c='blue', label='Non-outliers', alpha=0.6)

# Plot 10% low-density points in black
# plt.scatter(X[low_density_indices, 0], X[low_density_indices, 1], 
#             c='black', label='10% Low-density Points', alpha=0.6)

# Plot points classified as outliers based on the threshold in red
plt.scatter(X[outlier_scores > threshold, 0], X[outlier_scores > threshold, 1], 
            c='red', label='Threshold-based Outliers', alpha=0.6)

# Plot ignored points in orange
# plt.scatter(X[list(ignore_points), 0], X[list(ignore_points), 1], 
#             c='orange', label='Ignored Points', alpha=0.6)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Outliers Based on Threshold Value with Additional Categories')
plt.legend()
plt.grid(True)
plt.show()





# Final plot for outlier detection
# Convert the set to a list for indexing
outlier_indices = list(refined_outliers)

# Use the indices to extract points from the dataset
outlier_points = X[outlier_indices, :]

# Scatter plot of outlier points
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], label='inliers', c='blue')  # Plot all points in blue
plt.scatter(outlier_points[:, 0], outlier_points[:, 1], label='Outliers', c='red')  # Plot outliers in red

# Labeling the plot
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title("Plotting Refined Outliers")
plt.grid(True)
plt.show()

