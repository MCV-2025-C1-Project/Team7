from pathlib import Path
import pickle
import cv2

from descriptors import (
    compute_descriptors,
    hsv_histogram_concat,
    grayscale_histogram
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
from preprocess import preprocess_images, preprocess_images_laplacian


# main function that executes all:
# 1) load images in bgr
# 2) preprocess images
# 3) compute descriptors
# 4) retrieval
# 5) evaluate with mean average precision at K
# 6) print results
def main():
    TOPK = 10
    # 1) BBDD image paths
    pathlist = list(Path(Path(__file__).parent / "datasets" / "BBDD").glob("*.jpg"))

    # 1.1) Load BBDD images into a dictionary, key is the filename without extension and value is the image in bgr
    bbdd_images = {img_path.stem: cv2.imread(str(img_path)) for img_path in pathlist}

    # 2) Preprocess BBDD images (resize 256x256, color balance, contrast&brightness adjustment, smoothing)
    bbdd_images = preprocess_images(bbdd_images)

    # 3) Compute descriptors for BBDD images
    bbdd_rgb_descriptors = compute_descriptors(
        "bbdd",
        hsv_histogram_concat,
        bbdd_images,
        use_grayscale=False,
        save_as_pkl=True,
        overwrite_pkl=True,
    )

    # Repeate the same for Query descriptors of qsd1_w1
    pathlist = list(Path(Path(__file__).parent / "datasets" / "qsd1_w1").glob("*.jpg"))
    query_images = {img_path.stem: cv2.imread(str(img_path)) for img_path in pathlist}
    query_images = preprocess_images_laplacian(query_images)
    query_rgb_descriptors = compute_descriptors(
        "qsd1_w1",
        grayscale_histogram,
        query_images,
        use_grayscale=True,
        save_as_pkl=True,
        overwrite_pkl=True,
    )
    
    # Repeate the same for Query descriptors of qsd2_w1
    pathlist = list(Path(Path(__file__).parent / "datasets" / "qsd2_w1").glob("*.png"))
    query_gt_masks = [cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE) for img_path in pathlist]
    pathlist = list(Path(Path(__file__).parent / "datasets" / "qsd2_w1").glob("*.jpg"))
    query_images = {img_path.stem: cv2.imread(str(img_path)) for img_path in pathlist}
    """
    TODO:
        - Compute binary masks of query_images
        - loop through query_gt_masks and the computed binary masks calling metrics.BinaryMaskEvaluation
    """

    # Load ground truth correspondences
    gt = pickle.load(open("./datasets/qsd1_w1/gt_corresps.pkl", "rb"))

    # 4) Retrieval and 5) Evaluation with mAP@K
    print(f"Retrieval using {preprocess_images_laplacian.__name__} and Euclidean Distance")
    results = retrieval(
        bbdd_rgb_descriptors,
        query_rgb_descriptors,
        compute_euclidean_distance,
        top_k=TOPK,
    )
    print(f"mAP@K={TOPK}:", mean_average_precision_K(results, gt, K=TOPK))

    print(f"Retrieval using {preprocess_images_laplacian.__name__} and Manhattan Distance")
    results = retrieval(
        bbdd_rgb_descriptors,
        query_rgb_descriptors,
        compute_manhattan_distance,
        top_k=TOPK,
    )
    print(f"mAP@K={TOPK}:", mean_average_precision_K(results, gt, K=TOPK))

    print(f"Retrieval using {preprocess_images_laplacian.__name__} and X^2 Distance")
    results = retrieval(
        bbdd_rgb_descriptors, query_rgb_descriptors, compute_x2_distance, top_k=TOPK
    )
    # for query_index in range(len(query_rgb_descriptors)):
    #     print(f"{query_index}: {results[query_index]}")
    print(f"mAP@K={TOPK}:", mean_average_precision_K(results, gt, K=TOPK))

    print(f"Retrieval using {preprocess_images_laplacian.__name__} and Histogram Intersection")
    results = retrieval(
        bbdd_rgb_descriptors,
        query_rgb_descriptors,
        compute_histogram_intersection,
        top_k=TOPK,
    )
    print(f"mAP@K={TOPK}:", mean_average_precision_K(results, gt, K=TOPK))

    print(f"Retrieval using {preprocess_images_laplacian.__name__} and Hellinger Distance")
    results = retrieval(
        bbdd_rgb_descriptors,
        query_rgb_descriptors,
        compute_hellinger_distance,
        top_k=TOPK,
    )
    # for query_index in range(len(query_rgb_descriptors)):
    #     print(f"{query_index}: {results[query_index]}")
    print(f"mAP@K={TOPK}:", mean_average_precision_K(results, gt, K=TOPK))

    # # Compute for TESTSET and save as pkl
    # pathlist = list(Path(Path(__file__).parent / "datasets" / "qst1_w1").glob("*.jpg"))
    # test_images = {img_path.stem: cv2.imread(str(img_path)) for img_path in pathlist}
    # test_images = preprocess_images_laplacian(test_images)
    # test_rgb_descriptors = compute_descriptors(
    #     "qst1_w1",
    #     grayscale_histogram,
    #     test_images,
    #     use_grayscale=True,
    #     save_as_pkl=True,
    #     overwrite_pkl=True,
    # )
    # results = retrieval(
    #     bbdd_rgb_descriptors,
    #     test_rgb_descriptors,
    #     compute_manhattan_distance,
    #     top_k=TOPK,
    # )
    # sorted_results_listoflists = [
    #     [tup[1] for tup in results[query_index]] for query_index in range(len(test_rgb_descriptors))
    # ]
    # print(sorted_results_listoflists)
    # pickle.dump(
    #     sorted_results_listoflists, open("result.pkl", "wb")
    # )

if __name__ == "__main__":
    main()
