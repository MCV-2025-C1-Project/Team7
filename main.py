from pathlib import Path
import pickle
import cv2
from descriptors import (
    compute_descriptors,
    hsv_histogram_concat,
    hsv_hier_block_hist_concat_func,
    hsv_block_hist_concat_func,
)
from similarity import (
    compute_euclidean_distance,
    compute_manhattan_distance,
    compute_x2_distance,
    compute_histogram_intersection,
    compute_hellinger_distance,
)
from segmentation import compute_binary_mask_1, compute_binary_mask_2
from retrieval import retrieval
from preprocess import preprocess_images
from metrics import mean_average_precision_K, binary_mask_evaluation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


DISTANCE_FUNCTIONS = [
    compute_euclidean_distance,
    compute_manhattan_distance,
    compute_x2_distance,
    compute_histogram_intersection,
    compute_hellinger_distance,
]
TOPK = [1, 5, 10]  # Values of K for mAP@K evaluation


def compute_test_retrieval(
    bbdd_images, query_images, descriptor_function, distance_function, query_set_suffix
):
    """
    Function that computes bbdd & query images descriptors given descriptor function and does retrieval given a specific distance function and topk value.
    Saves the result in a pickle file as list of lists.
    """
    # Compute BBDD descriptors
    bbdd_descriptors = compute_descriptors(
        "bbdd",
        descriptor_function,
        bbdd_images,
        use_grayscale=False,
        save_as_pkl=False,
        overwrite_pkl=True,
    )

    # Repeate the same for test query descriptors
    query_descriptors = compute_descriptors(
        query_set_suffix,
        descriptor_function,
        query_images,
        use_grayscale=False,
        save_as_pkl=False,
        overwrite_pkl=True,
    )

    results = retrieval(
        bbdd_descriptors,
        query_descriptors,
        distance_function,
        top_k=10,
    )

    output_results = []
    for i in range(len(results)):
        output_results.append([res[1] for res in results[i]])
    print(
        f"Results for {descriptor_function.__name__} and {distance_function.__name__}:"
    )
    print(output_results)
    pickle.dump(
        output_results,
        open(
            f"results_{descriptor_function.__name__}_{distance_function.__name__}_{query_set_suffix}.pkl",
            "wb",
        ),
    )


def compute_retrieval_template(
    bbdd_images,
    query_images,
    descriptor_function,
    gt,
    query_set_suffix,
    visualize_output: bool = False,
):
    """
    Template function that computes bbdd & query images descriptors given descriptor function and does retrieval given a list of distance functions and topk values.
    Returns a dictionary with the results.
    """
    # Compute BBDD descriptors
    bbdd_descriptors = compute_descriptors(
        "bbdd",
        descriptor_function,
        bbdd_images,
        use_grayscale=False,
        save_as_pkl=True,
        overwrite_pkl=True,
    )

    # Repeate the same for Query descriptors of qsd1_w1
    query_descriptors = compute_descriptors(
        query_set_suffix,
        descriptor_function,
        query_images,
        use_grayscale=False,
        save_as_pkl=True,
        overwrite_pkl=True,
    )

    map_results = {}
    for k in TOPK:
        map_results[f"mAP@K={k}"] = {}
        for distance_function in DISTANCE_FUNCTIONS:
            results = retrieval(
                bbdd_descriptors,
                query_descriptors,
                distance_function,
                top_k=k,
            )
            if visualize_output:
                bbdd_trans = {int(k.split("_")[-1]): k for k, _ in bbdd_images.items()}
                query_trans = {int(k): k for k, _ in query_images.items()}
                for i in range(len(results)):
                    plt.figure(figsize=(15, 5))
                    plt.title(f"Query: {query_trans[i]}")
                    plt.subplot(1, k + 2, 1)
                    plt.imshow(query_images[query_trans[i]])
                    plt.title("Query")
                    plt.axis("off")
                    plt.subplot(1, k + 2, 2)
                    plt.imshow(bbdd_images[bbdd_trans[gt[i][0]]])
                    plt.title(f"GT: {bbdd_trans[gt[i][0]]}")
                    plt.axis("off")
                    for j in range(k):
                        plt.subplot(1, k + 2, j + 3)
                        retrieved_img_id = results[i][j][1]
                        plt.imshow(bbdd_images[bbdd_trans[retrieved_img_id]])
                        plt.title(f"Q{j + 1}: {bbdd_trans[retrieved_img_id]}")
                        plt.axis("off")
                    plt.show()
            map_results[f"mAP@K={k}"][distance_function.__name__] = (
                mean_average_precision_K(results, gt, K=k)
            )
    return map_results


def main():
    # test_week1_week2()
    best_of_each_week()


def best_of_each_week():
    week2_best_bins = [4, 4, 2]
    week2_best_grid = (9, 9)
    qsd1_pathlist = list(
        Path(Path(__file__).parent / "datasets" / "qsd1_w1").glob("*.jpg")
    )
    qsd2_pathlist = list(
        Path(Path(__file__).parent / "datasets" / "qsd2_w2").glob("*.jpg")
    )
    qsd2_masks_pathlist = list(
        Path(Path(__file__).parent / "datasets" / "qsd2_w2").glob("*.png")
    )
    qst1_pathlist = list(
        Path(Path(__file__).parent / "datasets" / "qst1_w2").glob("*.jpg")
    )
    qst2_pathlist = list(
        Path(Path(__file__).parent / "datasets" / "qst2_w2").glob("*.jpg")
    )
    bbdd_pathlist = list(
        Path(Path(__file__).parent / "datasets" / "BBDD").glob("*.jpg")
    )
    gt_qsd1_w1 = pickle.load(open("./datasets/qsd1_w1/gt_corresps.pkl", "rb"))
    gt_qsd2_w2 = pickle.load(open("./datasets/qsd2_w2/gt_corresps.pkl", "rb"))

    # 1.1) Load all images into a dictionary, key is the filename without extension and value is the image in bgr
    bbdd_images = {
        img_path.stem: cv2.imread(str(img_path)) for img_path in bbdd_pathlist
    }
    qsd1_images = {
        img_path.stem: cv2.imread(str(img_path)) for img_path in qsd1_pathlist
    }
    qsd2_images = {
        img_path.stem: cv2.imread(str(img_path)) for img_path in qsd2_pathlist
    }
    qst1_images = {
        img_path.stem: cv2.imread(str(img_path)) for img_path in qst1_pathlist
    }
    qst2_images = {
        img_path.stem: cv2.imread(str(img_path)) for img_path in qst2_pathlist
    }
    qsd2_gt_masks = {
        img_path.stem: cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        for img_path in qsd2_masks_pathlist
    }

    # WEEK 2 best segmentation method evaluation on qsd2_w2
    precision, recall, F1 = 0, 0, 0
    for name, img in qsd2_images.items():
        mask = compute_binary_mask_2(img)
        mask_gt = qsd2_gt_masks[name]

        mask_bool = mask > 0
        # masked_rgb = img * mask_bool[:, :, np.newaxis]

        cropped_img = img[np.ix_(mask_bool.any(1), mask_bool.any(0))]
        qsd2_images[name] = cropped_img

        metrics = binary_mask_evaluation(mask, mask_gt)

        precision += metrics["precision"]
        recall += metrics["recall"]
        F1 += metrics["F1"]

    avg_precision = precision / len(qsd2_images)
    avg_recall = recall / len(qsd2_images)
    avg_F1 = F1 / len(qsd2_images)
    print(
        f"WEEK 2: Binary Mask metrics: Precision: {avg_precision}. Recall: {avg_recall}. F1-measure: {avg_F1}."
    )

    output_masks_dir = Path("./output_masks")
    output_masks_dir.mkdir(parents=True, exist_ok=True)
    # WEEK 2 best segmentation method qst2_w2
    for name, img in qst2_images.items():
        mask = compute_binary_mask_2(img)
        cv2.imwrite(str(output_masks_dir / f"{name}.png"), mask * 255)
        mask_bool = mask > 0
        cropped_img = img[np.ix_(mask_bool.any(1), mask_bool.any(0))]
        qst2_images[name] = cropped_img

    lists_to_preprocess = [
        bbdd_images,
        qsd1_images,
        qsd2_images,
        qst1_images,
        qst2_images,
    ]
    for i in range(len(lists_to_preprocess)):
        lists_to_preprocess[i] = preprocess_images(lists_to_preprocess[i])

    # WEEK 1 best method execution
    map_results = compute_retrieval_template(
        bbdd_images, qsd1_images, hsv_histogram_concat, gt_qsd1_w1, "qsd1_w1"
    )
    print("WEEK 1: hsv_histogram_concat")
    print(pd.DataFrame(map_results))

    # WEEK 2 best method execution on qsd1_w1
    my_hsv_block_hist_concat = hsv_block_hist_concat_func(
        bins=week2_best_bins, grid=week2_best_grid
    )
    map_results = compute_retrieval_template(
        bbdd_images, qsd1_images, my_hsv_block_hist_concat, gt_qsd1_w1, "qsd1_w1"
    )
    print(
        f"WEEK 2: hsv_block_hist_concat bins={week2_best_bins} grid={week2_best_grid}, qsd1_w1"
    )
    print(pd.DataFrame(map_results))

    # WEEK 2 best method execution on qsd2_w2
    map_results = compute_retrieval_template(
        bbdd_images, qsd2_images, my_hsv_block_hist_concat, gt_qsd2_w2, "qsd2_w2"
    )
    print(
        f"WEEK 2: hsv_block_hist_concat bins={week2_best_bins} grid={week2_best_grid}, qsd2_w2"
    )
    print(pd.DataFrame(map_results))

    # WEEK 2 generate results for qst1_w2
    compute_test_retrieval(
        bbdd_images,
        qst1_images,
        my_hsv_block_hist_concat,
        compute_euclidean_distance,
        "qst1_w2",
    )
    # WEEK 2 generate results for qst2_w2
    compute_test_retrieval(
        bbdd_images,
        qst2_images,
        my_hsv_block_hist_concat,
        compute_euclidean_distance,
        "qst2_w2",
    )


# main function adapted for week 1&2 comparisons:
# 1) load all images in bgr
# 2) preprocess all images
# 3) compute week 1 method as reference point
# 4) compute week 2 methods with different parameters (nested loops, takes a long time to compute)
def test_week1_week2():
    run_block_histogram_concat = True
    run_hier_block_histogram_concat = False
    force_retrieval = True  # If True, forces recomputation of descriptors and retrieval even if result pkl files exist
    save_results = False  # If True, saves results of retrieval in method_bins_grids.pkl
    test_mode = False  # If True, only process a few images for quick testing & visualization purposes
    testing_bins = [[4, 4, 2]]
    testing_grids = [(9, 9)]
    testing_level_grids = [
        [(1, 1), (2, 2)],
        [(2, 2), (4, 4)],
        [(4, 4), (8, 8)],
        [(3, 3), (9, 9)],
    ]

    # Create results directory if it doesn't exist
    if save_results:
        Path("./df_results").mkdir(parents=True, exist_ok=True)

    # 1) Load all images paths
    if test_mode:
        qsd1_pathlist = list(
            Path(Path(__file__).parent / "datasets" / "qsd1_w1").glob("*.jpg")
        )[:5]
        qsd2_pathlist = list(
            Path(Path(__file__).parent / "datasets" / "qsd2_w2").glob("*.jpg")
        )[:5]
        qsd2_masks_pathlist = list(
            Path(Path(__file__).parent / "datasets" / "qsd2_w2").glob("*.png")
        )[:5]
        qst1_pathlist = list(
            Path(Path(__file__).parent / "datasets" / "qst1_w2").glob("*.jpg")
        )[:5]
        qst2_pathlist = list(
            Path(Path(__file__).parent / "datasets" / "qst2_w2").glob("*.jpg")
        )[:5]
    else:
        qsd1_pathlist = list(
            Path(Path(__file__).parent / "datasets" / "qsd1_w1").glob("*.jpg")
        )
        qsd2_pathlist = list(
            Path(Path(__file__).parent / "datasets" / "qsd2_w2").glob("*.jpg")
        )
        qsd2_masks_pathlist = list(
            Path(Path(__file__).parent / "datasets" / "qsd2_w2").glob("*.png")
        )
        qst1_pathlist = list(
            Path(Path(__file__).parent / "datasets" / "qst1_w2").glob("*.jpg")
        )
        qst2_pathlist = list(
            Path(Path(__file__).parent / "datasets" / "qst2_w2").glob("*.jpg")
        )
    bbdd_pathlist = list(
        Path(Path(__file__).parent / "datasets" / "BBDD").glob("*.jpg")
    )

    # Load ground truth correspondences
    gt_qsd1_w1 = pickle.load(open("./datasets/qsd1_w1/gt_corresps.pkl", "rb"))
    gt_qsd2_w2 = pickle.load(open("./datasets/qsd2_w2/gt_corresps.pkl", "rb"))

    # 1.1) Load all images into a dictionary, key is the filename without extension and value is the image in bgr
    bbdd_images = {
        img_path.stem: cv2.imread(str(img_path)) for img_path in bbdd_pathlist
    }
    qsd1_images = {
        img_path.stem: cv2.imread(str(img_path)) for img_path in qsd1_pathlist
    }
    qsd2_images = {
        img_path.stem: cv2.imread(str(img_path)) for img_path in qsd2_pathlist
    }
    qsd2_gt_masks = {
        img_path.stem: cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        for img_path in qsd2_masks_pathlist
    }

    precision, recall, F1 = 0, 0, 0
    for name, img in qsd2_images.items():
        mask = compute_binary_mask_2(img)
        mask_gt = qsd2_gt_masks[name]

        # plt.imshow(img)
        # plt.show()
        # plt.imshow(mask, cmap="grey")
        # plt.show()

        mask_bool = mask > 0
        # masked_rgb = img * mask_bool[:, :, np.newaxis]
        cropped_img = img[np.ix_(mask_bool.any(1), mask_bool.any(0))]
        qsd2_images[name] = cropped_img

        # plt.imshow(masked_rgb)
        # plt.show()

        metrics = binary_mask_evaluation(mask, mask_gt)

        precision += metrics["precision"]
        recall += metrics["recall"]
        F1 += metrics["F1"]

    avg_precision = precision / len(qsd2_images)
    avg_recall = recall / len(qsd2_images)
    avg_F1 = F1 / len(qsd2_images)
    print(
        f"WEEK 2: Binary Mask metrics: Precision: {avg_precision}. Recall: {avg_recall}. F1-measure: {avg_F1}."
    )

    # 2) Preprocess all images with the same preprocessing method (resize 256x256, color balance, contrast&brightness adjustment, smoothing)
    lists_to_preprocess = [bbdd_images, qsd1_images, qsd2_images]
    for i in range(len(lists_to_preprocess)):
        lists_to_preprocess[i] = preprocess_images(lists_to_preprocess[i])

    result_pkl_filename = Path("./df_results") / "qsd2_w2" / "hsv_histogram_concat.pkl"
    if result_pkl_filename.exists() and not force_retrieval:
        map_results = pickle.load(open(result_pkl_filename, "rb"))
        print("WEEK 1: hsv_histogram_concat")
        print(pd.DataFrame(map_results))
    else:
        # WEEK 1 method 1: hsv_histogram_concat x all distance metrics
        map_results = compute_retrieval_template(
            bbdd_images, qsd1_images, hsv_histogram_concat, gt_qsd1_w1, "qsd1_w1"
        )
        print("WEEK 1: hsv_histogram_concat")
        print(pd.DataFrame(map_results))
        print()
        if save_results:
            pickle.dump(map_results, open(result_pkl_filename, "wb"))

    # WEEK 2 methods 1 & 2: hsv_block_hist_concat & hsv_hier_block_hist_concat x all distance metrics
    for bins in testing_bins:
        for grid_idx, grid in enumerate(testing_grids):
            if run_block_histogram_concat:
                # START method hsv_block_hist_concat
                result_pkl_filename = (
                    Path("./df_results")
                    / "qsd2_w2"
                    / "qsd2_w2"
                    / "hsv_block_hist_concat_bins_{bins[0]}-{bins[1]}-{bins[2]}_grid_{grid[0]}-{grid[1]}.pkl"
                )
                if result_pkl_filename.exists() and not force_retrieval:
                    map_results = pickle.load(open(result_pkl_filename, "rb"))
                    print(f"WEEK 2: hsv_block_hist_concat bins={bins} grid={grid}")
                    print(pd.DataFrame(map_results))
                    continue
                my_hsv_block_hist_concat = hsv_block_hist_concat_func(
                    bins=bins, grid=grid
                )
                map_results = compute_retrieval_template(
                    bbdd_images,
                    qsd2_images,
                    my_hsv_block_hist_concat,
                    gt_qsd2_w2,
                    "qsd2_w2",
                    visualize_output=True,
                )
                print(f"WEEK 2: hsv_block_hist_concat bins={bins} grid={grid}")
                print(pd.DataFrame(map_results))
                print()
                if save_results:
                    pickle.dump(map_results, open(result_pkl_filename, "wb"))
                # END method hsv_block_hist_concat

            if run_hier_block_histogram_concat and grid_idx < len(testing_level_grids):
                # START method hsv_hier_block_hist_concat
                result_pkl_filename = (
                    Path("./df_results")
                    / "qsd2_w2"
                    / "qsd2_w2"
                    / "hsv_hier_block_hist_concat_bins_{bins[0]}-{bins[1]}-{bins[2]}_levels_grid_{testing_level_grids[grid_idx]}.pkl"
                )
                if result_pkl_filename.exists() and not force_retrieval:
                    map_results = pickle.load(open(result_pkl_filename, "rb"))
                    print(
                        f"WEEK 2: hsv_hier_block_hist_concat bins={bins} levels_grid={testing_level_grids[grid_idx]}"
                    )
                    print(pd.DataFrame(map_results))
                    continue
                my_hsv_hier_block_hist_concat = hsv_hier_block_hist_concat_func(
                    bins=bins, levels_grid=testing_level_grids[grid_idx]
                )
                map_results = compute_retrieval_template(
                    bbdd_images,
                    qsd2_images,
                    my_hsv_hier_block_hist_concat,
                    gt_qsd2_w2,
                    "qsd2_w2",
                )
                print(
                    f"WEEK 2: hsv_hier_block_hist_concat bins={bins} levels_grid={testing_level_grids[grid_idx]}"
                )
                print(pd.DataFrame(map_results))
                print()
                if save_results:
                    pickle.dump(map_results, open(result_pkl_filename, "wb"))
                # END method hsv_hier_block_hist_concat


if __name__ == "__main__":
    main()
