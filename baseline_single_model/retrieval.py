import numpy as np

def find_similar_images(feature, reference_features):
    """
    Find the index of the most similar reference feature to the given feature
    based on Euclidean distance.
    """
    distances = np.linalg.norm(reference_features - feature, axis=1)
    similar_index = np.argmin(distances)
    return similar_index