


import math
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn import metrics
import os


def find_local_density(distances, k, dimension):
    density_sum = 0
    for dist in distances:
        density_sum += ((1 / (2 * math.pi)) ** dimension) * math.exp(-0.5 * ((dist) ** 2))
    return density_sum / k

def outlier_detection(X, k, dimension, density_percentile=10):
    # Step 1: Calculate k-NN graph
    knn_graph = NearestNeighbors(n_neighbors=k+1).fit(X)
    # alpha = 0.05
    # Step 2: Initialize outlier scores
    # outlier_scores = np.ones(len(X))
    outlier_scores = np.zeros(len(X))
    alpha = 0.05
    # Step 3: Calculate local density for all points
    local_densities = []
    temp_local_densities = []
    distance_matrix = []
    neighbor_index_matrix = []
    for i, x in enumerate(X):
        distances, neighbors_indices= knn_graph.kneighbors([x], return_distance=True)
        distance_matrix.append(distances[0])
        neighbor_index_matrix.append(neighbors_indices[0])
        p_x = find_local_density(distances[0], k, dimension)
        local_densities.append(p_x)
        temp_local_densities.append((p_x, i))
    
    # Step 4: Sort densities and select the lowest density points (bottom 10%)
    temp_local_densities.sort(key=lambda x: x[0])  # Sort by local density value
    num_low_density_points = int(len(X) * (density_percentile / 100.0))  # 10% of the points

    # Get the indices of low density points
    low_density_indices = []
    low_density_indices_set = set()
    for (_, index) in temp_local_densities[0:num_low_density_points]:
        low_density_indices.append(index)
        low_density_indices_set.add(index)
    matrix = []
    # Step 5: Calculate outlier scores for low-density points using high-density iterations
    for i in low_density_indices:
        
        # Step 5.1: Calculate local density for the current point
        p_x = local_densities[i]
        curr_max_density = p_x
        neighbors_indices = neighbor_index_matrix[i]
        first_neighbor_density = p_x
        # Step 5.2: Iteratively update density and calculate outlier score
        while (True):
            max_neighbor_density = float('-inf')
            max_density_index = 0
            for j, index in enumerate(neighbors_indices):
                if local_densities[index] > max_neighbor_density:
                    max_neighbor_density = local_densities[index]
                    max_density_index = index
                # if local_densities[index] < max_neighbor_density:
                #     max_neighbor_density = local_densities[index]
                #     max_density_index = index

            if curr_max_density < max_neighbor_density:
                if first_neighbor_density == p_x :
                    first_neighbor_density = max_neighbor_density
                neighbors_indices = neighbor_index_matrix[max_density_index]
                curr_max_density = max_neighbor_density
                if max_density_index not in low_density_indices_set :
                    break
            else :
                break
        # Calculating COF or outlier scores for low-density points
        # outlier_scores[i] = ((1-alpha)*first_neighbor_density + alpha*curr_max_density)/ p_x
        outlier_scores[i] = ((1-alpha)*curr_max_density + alpha*p_x)/ p_x
        # outlier_scores[i] = p_x / curr_max_density
        matrix.append((i,local_densities[i],outlier_scores[i]))
        # outlier_scores[i] = p_x / curr_max_density

    return low_density_indices,local_densities,outlier_scores,matrix

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
# file_to_load = "DS02.txt"
# file_to_load = "DS11.txt"
file_to_load = "DS04.txt"
# file_to_load = "DS03.txt"
df = load_data(file_to_load)


X = df
new_X = X[:, 0:2]

# Set parameters
k = 10
# m = 10
dimension = X.shape[1] - 1

# Detect outliers using 10% least dense points for high-density iterations
low_density_indices,local_densities,outlier_scores,matrix = outlier_detection(new_X, k, dimension, density_percentile=10)

# Identify threshold based on outlier_scores
a = 2.5
b = 1
c = 3

new_outlier_scores = []
for i,score in enumerate(outlier_scores) :
    if i in low_density_indices :
        new_outlier_scores.append(score)
# median of only 10% of data points jinka hmne outlier score nikala h
median_of_OS = np.median(new_outlier_scores) 

# median_of_OS_minus_median = np.median(abs(new_outlier_scores - median_of_OS))
# median_of_OS_minus_median = median_of_OS + \
                        # np.mean((new_outlier_scores - median_of_OS)**2)/np.mean(new_outlier_scores)

## median of all data points outlier score 
# median_of_OS = np.median(outlier_scores)
# median_of_OS_minus_median = np.median(abs(outlier_scores - median_of_OS))


# mad = b * median_of_OS_minus_median
# threshold = median_of_OS + a * mad
threshold = median_of_OS + np.mean((new_outlier_scores - median_of_OS)**2)/np.mean(new_outlier_scores)

# threshold = median_of_OS + c * median_of_OS_minus_median

# Identify outliers based on the threshold
outliers = np.where(outlier_scores > threshold)[0]

print(len(outliers))
# Generate the 'out' array for outliers
S = len(X)
out = np.zeros(S)
# for i in range(S):
#     if i in outliers:
#         out[i] = 1

new_matrix = []

for i in range(S):
    if i not in outliers and X[i][2]==1:
        # print(x[0]," ",x[1]," ",x[2])
        new_matrix.append([i,local_densities[i],outlier_scores[i],1,0])
        out[i] = 0


for x in matrix:
    if x[0] in outliers and X[x[0]][2]==0:
        # print(x[0]," ",x[1]," ",x[2])
        new_matrix.append([x[0],x[1],x[2],0,1])
        out[i] = 1

for x in matrix:
    if x[0] in outliers and X[x[0]][2]==1:
        # print(x[0]," ",x[1]," ",x[2])
        new_matrix.append([x[0],x[1],x[2],1,1])
        out[i] = 1


# Convert the matrix list into a DataFrame
df_matrix = pd.DataFrame(new_matrix, columns=['Index', 'Density', 'OutlierScore','Actual','Predicted'])

# Save the DataFrame to an Excel file
output_file = file_to_load + "matrix_output.xlsx"
df_matrix.to_excel(output_file, index=False)

print(f"Matrix data has been saved to {output_file}")




# Assuming the actual labels (Y) are in the third column of the dataset (adjust accordingly)
Y = X[:, 2]  # Replace this with the correct index if needed

#plot actual outlier data
# Plot the data (Class 0 and Class 1 points)
plt.figure(figsize=(10, 6))

# Plot class 0 points (X[i][2] == 0)
plt.scatter(X[X[:, 2] == 0, 0], X[X[:, 2] == 0, 1], 
            c='blue', label='Class 0', alpha=0.6)

# Plot class 1 points (X[i][2] == 1)
plt.scatter(X[X[:, 2] == 1, 0], X[X[:, 2] == 1, 1], 
            c='red', label='Class 1', alpha=0.6)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot based on actual outlier data (Class 0 or 1)')
plt.legend()
plt.grid(True)
plt.show()

#plot least density data

# Plot the data (low-density and non-low-density points)
plt.figure(figsize=(10, 6))

# Plot non-low-density points in blue
plt.scatter(
    X[~np.isin(np.arange(X.shape[0]), low_density_indices), 0], 
    X[~np.isin(np.arange(X.shape[0]), low_density_indices), 1], 
    c='blue', label='High-density points', alpha=0.6
)

# Plot low-density points in red
plt.scatter(
    X[low_density_indices, 0], 
    X[low_density_indices, 1], 
    c='red', label='Low-density points', alpha=0.6
)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot of Low and High Density Points')
plt.legend()
plt.grid(True)
plt.show()


# Plot the data (non-outliers and outliers) actual model
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

# Compute ROC curve and AUC
fpr, tpr, _ = metrics.roc_curve(Y, out)
auc = metrics.roc_auc_score(Y, out)
    
# Plot ROC curve
plt.plot(fpr, tpr, label=f"Binary Class AUC={auc:.2f}")

# Create plot for ROC curves of all classes
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curves')
plt.legend(loc=4)
plt.savefig('roc_curve.png', dpi=2400)  # Save with high DPI
plt.show()

# Optionally, print AUC for each class

print(f"AUC : {auc}")



