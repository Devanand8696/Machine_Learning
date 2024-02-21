import csv
import numpy as np
import matplotlib.pyplot as plt

# Read the dataset from "cancer.csv" and include all columns from index 2 to the end
data = []
with open('cancer.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)  # Skip the header row
    for row in reader:
        data.append([float(x) for x in row[2:]])  # Include all columns from index 2 to the end

data = np.array(data)

# Normalize the data (scaling to a range between 0 and 1)
def normalize_data(data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data

normalized_data = normalize_data(data)

# Define a function to compute the Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Define the DBSCAN algorithm
def dbscan(data, epsilon, min_samples):
    n = len(data)
    labels = np.zeros(n, dtype=int)
    cluster_id = 0

    for i in range(n):
        if labels[i] != 0:
            continue

        neighbors = [j for j in range(n) if euclidean_distance(data[i], data[j]) <= epsilon]

        if len(neighbors) < min_samples:
            labels[i] = -1  # Mark as noise
        else:
            cluster_id += 1
            labels[i] = cluster_id
            expand_cluster(data, labels, i, neighbors, cluster_id, epsilon, min_samples)

    return labels

def expand_cluster(data, labels, core_point_index, neighbors, cluster_id, epsilon, min_samples):
    for neighbor in neighbors:
        if labels[neighbor] == 0:
            labels[neighbor] = cluster_id
            neighbor_neighbors = [j for j in range(len(data)) if euclidean_distance(data[neighbor], data[j]) <= epsilon]
            if len(neighbor_neighbors) >= min_samples:
                neighbors.extend(neighbor_neighbors)

# Set DBSCAN parameters
parameters = [
    {"epsilon": 0.2, "min_samples": 6},
    {"epsilon": 0.5, "min_samples": 6},
    {"epsilon": 0.2, "min_samples": 3}
]

# Plot clusters for different parameter settings
plt.figure(figsize=(5, 15))  # Adjust the figure size for vertical layout

for i, params in enumerate(parameters):
    epsilon = params["epsilon"]
    min_samples = params["min_samples"]

    # Perform DBSCAN clustering
    cluster_labels = dbscan(normalized_data, epsilon, min_samples)

    plt.subplot(3, 1, i + 1)  # Separate rows, one column
    plt.title(f"Eps = {epsilon}, MinPoints = {min_samples}")
    
    unique_labels = np.unique(cluster_labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        if label == -1:
            plt.scatter(normalized_data[cluster_labels == label][:, 0], normalized_data[cluster_labels == label][:, 1],
                        c='k', marker='.', label=f'Noise ({np.sum(cluster_labels == label)} points)')
        else:
            plt.scatter(normalized_data[cluster_labels == label][:, 0], normalized_data[cluster_labels == label][:, 1],
                        c=color, marker='.', label=f'Cluster {label} ({np.sum(cluster_labels == label)} points)')

    plt.tight_layout()
    plt.xlabel("radius_mean")
    plt.ylabel("texture_mean")
    plt.legend()

plt.show()
