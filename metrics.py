def mean_average_precision_K(
    results: dict[int, list[tuple[float, int]]], gt: list[list[int]], K: int = 1
):
    """
    Computes the mean average precision at K.
    Args:
        results: A dictionary where keys are query indices and values are lists of retrieved image indices.
        gt: A dictionary where keys are query indices and values are sets of ground truth relevant image indices.
    Returns:
        The mean average precision at K.
    """
    ap_sum = 0.0
    num_queries = len(results)

    for query_index, retrieved in results.items():
        relevant = gt[query_index]
        if not relevant:
            continue

        num_retrieved_relevant = 0
        precision_sum = 0.0

        for k, tuple_retrieval in enumerate(retrieved[:K], start=1):
            img_index = tuple_retrieval[1]
            if img_index in relevant:
                num_retrieved_relevant += 1
                precision_sum += num_retrieved_relevant / k

        if num_retrieved_relevant > 0:
            ap_sum += precision_sum / len(relevant)

    return ap_sum / num_queries if num_queries > 0 else 0.0
