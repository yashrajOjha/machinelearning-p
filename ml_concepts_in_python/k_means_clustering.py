import numpy as np
def get_distances(centroids, np_points):
    distances = []
    for centroid in centroids:
        c_ = np.array(centroid)
        dist = np.sqrt(np.sum((np_points-c_)**2, axis=1))
        distances.append(dist.tolist())
    return distances

def get_nearest_centroid(distances):
    clusters = np.argmin(np.array(distances), axis=0)
    return clusters

def k_means_clustering(points: list[tuple[float, float]], k: int, initial_centroids: list[tuple[float, float]], max_iterations: int) -> list[tuple[float, float]]:
    centroids = initial_centroids
    np_points = np.array(points)
    for _ in range(max_iterations):
        distances = get_distances(centroids, points)
        cluster_ids = get_nearest_centroid(distances)
        previous_centroids = centroids
        for i in range(k):
            regrouped = np_points[cluster_ids == i]
            centroids[i]= np.mean(regrouped, axis=0)
        if previous_centroids==centroids:
            break
    
    final_centroids = []
    for c in centroids:
        final_centroids.append(tuple(c.tolist()))
        
    return final_centroids