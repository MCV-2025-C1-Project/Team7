from collections.abc import Callable
import numpy as np


def retrieval(
    bbdd_descriptors: dict[int, list[np.ndarray]],
    query_decriptors: dict[int, list[np.ndarray]],
    similarity: Callable[[np.ndarray, np.ndarray], float],
    top_k: int = 5,
) -> dict[int, list[list[tuple[float, int]]]]:
    """
    Perform image retrieval by comparing query descriptors against a database of descriptors using a specified similarity function.
    Args:
        bbdd_descriptors: A dictionary mapping image indices to their descriptors in the database.
        query_decriptors: A dictionary mapping image indices to their descriptors for the queries.
        similarity: A callable function that takes two descriptors and returns a similarity score.
        top_k: The number of top similar images to retrieve for each query.
    Returns:
        A dictionary mapping each query index to a list of tuples containing the top_k most similar database image indices and their similarity scores.
    """

    # Compute similarities
    similarities = {}
    for query_index, query_descriptor in query_decriptors.items():
        similarities[query_index] = []
        for single_query_descriptor in query_descriptor:
            sims = {}
            for db_index, db_descriptor in bbdd_descriptors.items():
                distance = similarity(single_query_descriptor, db_descriptor[0])
                sims[db_index] = distance
            similarities[query_index].append(sims)

    # Get top K results
    top_results = {}
    for query_index, subsims in similarities.items():
        top_results[query_index] = []
        for distances in subsims:
            sorted_distances = sorted([(v, k) for (k, v) in distances.items()])[:top_k]
            top_results[query_index].append(sorted_distances)

    return top_results
