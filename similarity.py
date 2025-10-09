import numpy as np


def compute_euclidean_distance(
    descriptor1: np.ndarray, descriptor2: np.ndarray
) -> float:
    """
    Compute the Euclidean distance between two descriptors.
    Args:
        descriptor1: A 1D numpy array representing the first descriptor.
        descriptor2: A 1D numpy array representing the second descriptor.
    Returns:
        A float representing the Euclidean distance between the two descriptors.
    """
    return float(np.linalg.norm(descriptor1 - descriptor2))


def compute_manhattan_distance(
    descriptor1: np.ndarray, descriptor2: np.ndarray
) -> float:
    """
    Compute the Manhattan distance between two descriptors.
    Args:
        descriptor1: A 1D numpy array representing the first descriptor.
        descriptor2: A 1D numpy array representing the second descriptor.
    Returns:
        A float representing the Manhattan distance between the two descriptors.
    """
    return np.sum(np.abs(descriptor1 - descriptor2))


def compute_x2_distance(descriptor1: np.ndarray, descriptor2: np.ndarray) -> float:
    """
    Compute the X^2 distance between two descriptors.
    Args:
        descriptor1: A 1D numpy array representing the first descriptor.
        descriptor2: A 1D numpy array representing the second descriptor.
    Returns:
        A float representing the X^2 distance between the two descriptors.
    """
    return float(np.sum(((descriptor1 - descriptor2) ** 2) / (descriptor1 + descriptor2 + 1e-10)))


def compute_histogram_intersection(
    descriptor1: np.ndarray, descriptor2: np.ndarray
) -> float:
    """
    Compute the histogram intersection between two descriptors.
    Args:
        descriptor1: A 1D numpy array representing the first descriptor.
        descriptor2: A 1D numpy array representing the second descriptor.
    Returns:
        A float representing the histogram intersection between the two descriptors.
    """
    return float(np.sum(np.minimum(descriptor1, descriptor2)))


def compute_hellinger_distance(
    descriptor1: np.ndarray, descriptor2: np.ndarray
) -> float:
    """
    Compute the Hellinger distance between two descriptors.
    Args:
        descriptor1: A 1D numpy array representing the first descriptor.
        descriptor2: A 1D numpy array representing the second descriptor.
    Returns:
        A float representing the Hellinger distance between the two descriptors.
    """
    return float(np.sqrt(1 - np.sum(np.sqrt(descriptor1 * descriptor2))))
