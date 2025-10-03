from pathlib import Path
import pickle
from PIL import Image
import numpy as np

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
from prediction.base import mean_average_precision_K


def main():
    # BBDD descriptors
    pathlist = list(Path(Path(__file__).parent / "datasets" / "BBDD").glob("*.jpg"))
    bbdd_images = {img_path.stem: np.array(Image.open(img_path)) for img_path in pathlist}
    
    bbdd_grayscale_descriptors = compute_descriptors(
        "bbdd",
        grayscale_histogram, bbdd_images, save_as_pkl=False
    )
    bbdd_rgb_descriptors = compute_descriptors(
        "bbdd", concat_rgb_histogram, bbdd_images, use_grayscale=False, save_as_pkl=True, overwrite_pkl=True
    )

    # Query descriptors
    pathlist = list(Path(Path(__file__).parent / "datasets" / "qsd1_w1").glob("*.jpg"))
    query_images = {img_path.stem: np.array(Image.open(img_path)) for img_path in pathlist}
    query_grayscale_descriptors = compute_descriptors(
        "qsd1_w1", grayscale_histogram, query_images, save_as_pkl=False
    )
    query_rgb_descriptors = compute_descriptors(
        "qsd1_w1", concat_rgb_histogram, query_images, use_grayscale=False, save_as_pkl=True, overwrite_pkl=True
    )

    gt = pickle.load(open("./descriptor/qsd1_w1/gt.pkl", "rb"))

    print("Retrieval using Grayscale Histograms and Euclidean Distance")
    results = retrieval(
        bbdd_grayscale_descriptors,
        query_grayscale_descriptors,
        compute_euclidean_distance,
    )
    mean_average_precision_K(results, gt, K=5)
    results = sorted(results.items())
    for query_index, retrieved in results:
        print(f"{query_index}: {retrieved}")

    print("Retrieval using Grayscale Histograms and Manhattan Distance")
    results = retrieval(
        bbdd_grayscale_descriptors,
        query_grayscale_descriptors,
        compute_manhattan_distance,
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
        bbdd_grayscale_descriptors,
        query_grayscale_descriptors,
        compute_histogram_intersection,
    )
    results = sorted(results.items())
    for query_index, retrieved in results:
        print(f"{query_index}: {retrieved}")

    print("Retrieval using Grayscale Histograms and Hellinger Distance")
    results = retrieval(
        bbdd_grayscale_descriptors,
        query_grayscale_descriptors,
        compute_hellinger_distance,
    )
    results = sorted(results.items())
    for query_index, retrieved in results:
        print(f"{query_index}: {retrieved}")


if __name__ == "__main__":
    main()
