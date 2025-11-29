import numpy as np

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x-y)**2), axis=1)
    return sum([(p1-p2)**2 for p1, p2 in zip(x, y)])**1/2

def finding_neighbours(train_samples, train_labels, query, k):
    distances = []
    for ind, train_sample in enumerate(train_samples):
        dist = euclidean_distance(query, train_sample)
        distances.append((ind, dist))
    
    distances.sort(key=lambda x: x[1])
    
    neighbours = [ind for ind, dist in distances[:k]]
    return neighbours

def predict_regression(train_samples, train_labels, query, k):
    """
    For regression: return the mean value of k nearest neighbors
    """
    neighbours = finding_neighbours(train_samples, train_labels, query, k)

    k_nearest_values = [train_labels[ind] for ind in neighbours]
    
    # For regression: return mean
    prediction = np.mean(k_nearest_values)
    return prediction

def predict_classification(train_samples, train_labels, query, k):
    """
    For classification: return the most common label of k nearest neighbors
    """
    neighbours = finding_neighbours(train_samples, train_labels, query, k)
    
    k_nearest_labels = [train_labels[ind] for ind in neighbours]
    
    label_counts = {}
    for label in k_nearest_labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    predicted_class = max(label_counts, key=label_counts.get)
    return predicted_class