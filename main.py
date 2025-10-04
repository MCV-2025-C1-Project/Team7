from pathlib import Path
import pickle
import cv2

from descriptors import (
    rgb_hist_hellinger,
    compute_descriptors,
)
from similarity import (
    compute_euclidean_distance,
    compute_manhattan_distance,
    compute_x2_distance,
    compute_histogram_intersection,
    compute_hellinger_distance,
)
from retrieval import retrieval
from metrics import mean_average_precision_K
from preprocess import preprocess_images


# main function that executes all:
# 1) load images in bgr
# 2) preprocess images
# 3) compute descriptors
# 4) retrieval
# 5) evaluate with mean average precision at K
# 6) print results
def main():
    TOPK = 2
    # 1) BBDD image paths
    pathlist = list(Path(Path(__file__).parent / "datasets" / "BBDD").glob("*.jpg"))

    # 1.1) Load BBDD images into a dictionary, key is the filename without extension and value is the image in bgr
    bbdd_images = {img_path.stem: cv2.imread(str(img_path)) for img_path in pathlist}

    # 2) Preprocess BBDD images (resize 256x256, color balance, contrast&brightness adjustment, smoothing)
    # TODO: Check if preprocessing is doing all the steps correctly
    bbdd_images = preprocess_images(bbdd_images)

    # 3) Compute descriptors for BBDD images
    # TODO: Check if the histograms are computed correctly
    bbdd_rgb_descriptors = compute_descriptors(
        "bbdd",
        rgb_hist_hellinger,
        bbdd_images,
        use_grayscale=False,
        save_as_pkl=True,
        overwrite_pkl=True,
    )

    # Repeate the same for Query descriptors
    pathlist = list(Path(Path(__file__).parent / "datasets" / "qsd1_w1").glob("*.jpg"))
    query_images = {img_path.stem: cv2.imread(str(img_path)) for img_path in pathlist}
    query_rgb_descriptors = compute_descriptors(
        "qsd1_w1",
        rgb_hist_hellinger,
        query_images,
        use_grayscale=False,
        save_as_pkl=True,
        overwrite_pkl=True,
    )

    # Load ground truth correspondences
    gt = pickle.load(open("./datasets/qsd1_w1/gt_corresps.pkl", "rb"))

    # 4) Retrieval and 5) Evaluation with mAP@K
    # TODO: Check and compare what is different from cv2 compareHist and our implementation
    # Maybe I'm doing something wrong in the comparison?
    print("Retrieval using RGB Histograms + Hellinger normalization and X^2 Distance")
    results = retrieval(
        bbdd_rgb_descriptors, query_rgb_descriptors, compute_x2_distance, top_k=TOPK
    )
    for query_index in range(len(query_rgb_descriptors)):
        print(f"{query_index}: {results[query_index]}")
    print("mAP@K:", mean_average_precision_K(results, gt, K=TOPK))

    print(
        "Retrieval using RGB Histograms + Hellinger normalization and Hellinger Distance"
    )
    results = retrieval(
        bbdd_rgb_descriptors,
        query_rgb_descriptors,
        compute_hellinger_distance,
        top_k=TOPK,
    )
    for query_index in range(len(query_rgb_descriptors)):
        print(f"{query_index}: {results[query_index]}")
    print("mAP@K:", mean_average_precision_K(results, gt, K=TOPK))


if __name__ == "__main__":
    main()
