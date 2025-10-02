from pathlib import Path

from descriptor.base import (
    grayscale_histogram,
    concat_rgb_histogram,
    compute_descriptors,
)
from similarity.base import (
    compute_euclidean_distance,
    compute_manhattan_distance,
    compute_x2_distance,
    compute_histogram_intersection,
    compute_hellinger_distance,
)
from retrieval.base import retrieval


def main():
    # BBDD descriptors
    pathlist = list(Path("./descriptor/BBDD").glob("*.jpg"))
    bbdd_grayscale_descriptors = compute_descriptors(grayscale_histogram, pathlist, save_as_pkl=True)
    # bbdd_rgb_descriptors = compute_descriptors(concat_rgb_histogram, pathlist, save_as_pkl=True)

    # Query descriptors
    pathlist = list(Path("./descriptor/qsd1_w1").glob("*.jpg"))
    query_grayscale_descriptors = compute_descriptors(grayscale_histogram, pathlist, save_as_pkl=True)
    # query_rgb_descriptors = compute_descriptors(concat_rgb_histogram, pathlist, save_as_pkl=True)

    print("Retrieval using Grayscale Histograms and Euclidean Distance")
    results = retrieval(
        bbdd_grayscale_descriptors,
        query_grayscale_descriptors,
        compute_euclidean_distance,
    )
    results = sorted(results.items())
    for query_index, retrieved in results:
        print(f"{query_index}: {retrieved}")

    print("Retrieval using Grayscale Histograms and Manhattan Distance")
    results = retrieval(
        bbdd_grayscale_descriptors, query_grayscale_descriptors, compute_manhattan_distance
    )
    results = sorted(results.items())
    for query_index, retrieved in results:
        print(f"{query_index}: {retrieved}")

    print("Retrieval using Grayscale Histograms and X^2 Distance")
    results = retrieval(
        bbdd_grayscale_descriptors, query_grayscale_descriptors, compute_x2_distance
    )
    results = sorted(results.items())
    for query_index, retrieved in results:
        print(f"{query_index}: {retrieved}")

    print("Retrieval using Grayscale Histograms and Histogram Intersection")
    results = retrieval(
        bbdd_grayscale_descriptors, query_grayscale_descriptors, compute_histogram_intersection
    )
    results = sorted(results.items())
    for query_index, retrieved in results:
        print(f"{query_index}: {retrieved}")

    print("Retrieval using Grayscale Histograms and Hellinger Distance")
    results = retrieval(
        bbdd_grayscale_descriptors, query_grayscale_descriptors, compute_hellinger_distance
    )
    results = sorted(results.items())
    for query_index, retrieved in results:
        print(f"{query_index}: {retrieved}")


if __name__ == "__main__":
    main()
