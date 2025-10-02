import numpy as np


def compute_euclidean_distance(
    descriptor1: np.ndarray, descriptor2: np.ndarray
) -> float:
    '''
    Compute the Euclidean distance between two descriptors.
    Args:
        descriptor1: A 1D numpy array representing the first descriptor.
        descriptor2: A 1D numpy array representing the second descriptor.
    Returns:
        A float representing the Euclidean distance between the two descriptors.
    '''
    
    acum = 0.0
    for h1, h2 in zip(descriptor1, descriptor2):
        acum += (h1 - h2) ** 2
    return np.sqrt(acum)


def compute_manhattan_distance(
    descriptor1: np.ndarray, descriptor2: np.ndarray
) -> float:
    '''
    Compute the Manhattan distance between two descriptors.
    Args:
        descriptor1: A 1D numpy array representing the first descriptor.
        descriptor2: A 1D numpy array representing the second descriptor.
    Returns:
        A float representing the Manhattan distance between the two descriptors.
    '''

    acum = 0.0
    for h1, h2 in zip(descriptor1, descriptor2):
        acum += abs(h1 - h2)
    return acum


def compute_x2_distance(descriptor1: np.ndarray, descriptor2: np.ndarray) -> float:
    '''
    Compute the X^2 distance between two descriptors.
    Args:
        descriptor1: A 1D numpy array representing the first descriptor.
        descriptor2: A 1D numpy array representing the second descriptor.
    Returns:
        A float representing the X^2 distance between the two descriptors.
    '''

    acum = 0.0
    for h1, h2 in zip(descriptor1, descriptor2):
        if h1 + h2 != 0:
            acum += ((h1 - h2) ** 2) / (h1 + h2)
    return acum


def compute_histogram_intersection(
    descriptor1: np.ndarray, descriptor2: np.ndarray
) -> float:
    '''
    Compute the histogram intersection between two descriptors.
    Args:
        descriptor1: A 1D numpy array representing the first descriptor.
        descriptor2: A 1D numpy array representing the second descriptor.
    Returns:
        A float representing the histogram intersection between the two descriptors.
    '''

    acum = 0.0
    for h1, h2 in zip(descriptor1, descriptor2):
        acum += min(h1, h2)
    return acum


def compute_hellinger_distance(
    descriptor1: np.ndarray, descriptor2: np.ndarray
) -> float:
    '''
    Compute the Hellinger distance between two descriptors.
    Args:
        descriptor1: A 1D numpy array representing the first descriptor.
        descriptor2: A 1D numpy array representing the second descriptor.
    Returns:
        A float representing the Hellinger distance between the two descriptors.
    '''

    acum = 0.0
    for h1, h2 in zip(descriptor1, descriptor2):
        acum += (np.sqrt(h1) - np.sqrt(h2)) ** 2
    return (1 / np.sqrt(2)) * np.sqrt(acum)
