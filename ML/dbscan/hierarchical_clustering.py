import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv("cancer.csv")

# Extract relevant features for clustering (excluding "id" and "diagnosis")
X = data.drop(columns=["id", "diagnosis"])

# Normalize the data using StandardScaler
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Number of clusters
num_clusters = 2

# Define the single linkage function
def single_linkage(cluster1, cluster2):
    min_distance = np.inf
    for point1 in cluster1:
        for point2 in cluster2:
            distance = np.linalg.norm(point1 - point2)
            min_distance = min(min_distance, distance)
    return min_distance

# Define the complete linkage function
def complete_linkage(cluster1, cluster2):
    max_distance = -np.inf
    for point1 in cluster1:
        for point2 in cluster2:
            distance = np.linalg.norm(point1 - point2)
            max_distance = max(max_distance, distance)
    return max_distance

# Define the average linkage function
def average_linkage(cluster1, cluster2):
    total_distance = 0
    count = 0
    for point1 in cluster1:
        for point2 in cluster2:
            distance = np.linalg.norm(point1 - point2)
            total_distance += distance
            count += 1
    return total_distance / count

# Function to perform hierarchical clustering
def hierarchical_clustering(X, num_clusters, linkage_func):
    clusters = [[point] for point in X]

    while len(clusters) > num_clusters:
        min_distance = np.inf
        merge_indices = (-1, -1)

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distance = linkage_func(clusters[i], clusters[j])
                if distance < min_distance:
                    min_distance = distance
                    merge_indices = (i, j)

        i, j = merge_indices
        clusters[i].extend(clusters[j])
        del clusters[j]

    return clusters

# Perform hierarchical clustering with different linkages
linkages = ["single", "complete", "average"]
plt.figure(figsize=(15, 5))

for i, linkage in enumerate(linkages):
    plt.subplot(1, 3, i + 1)
    plt.title(f"Linkage={linkage}")
    
    if linkage == "single":
        linkage_func = single_linkage
    elif linkage == "complete":
        linkage_func = complete_linkage
    elif linkage == "average":
        linkage_func = average_linkage
    
    clusters = hierarchical_clustering(X_normalized, num_clusters, linkage_func)

    # Plot clusters
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    for j, cluster in enumerate(clusters):
        cluster = np.array(cluster)
        plt.scatter(cluster[:, 0], cluster[:, 1], c=colors[j], label=f'Cluster {j + 1}')

    plt.xlabel('Normalized Feature 1')
    plt.ylabel('Normalized Feature 2')
    plt.legend()

plt.tight_layout()
plt.show()
