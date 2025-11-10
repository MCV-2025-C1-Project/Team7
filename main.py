import pickle
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from descriptors import (
    compute_descriptors,
    dct_block_descriptor_func,
    glcm_block_descriptor_func,
    hsv_block_hist_concat_func,
    hsv_histogram_concat,
)
from keypoints import (
    calculate_matches,
    compute_local_descriptors,
    harris_corner_detection_func,
    harris_laplacian_detection_func,
    keypoint_retrieval,
)
from metrics import binary_mask_evaluation, mean_average_precision_K
from preprocess import preprocess_images
from retrieval import retrieval
from segmentation import get_crops_from_gt_mask, get_mask_and_crops_refined
from similarity import (
    compute_euclidean_distance,
    compute_hellinger_distance,
    compute_histogram_intersection,
    compute_manhattan_distance,
    compute_x2_distance,
)

DISTANCE_FUNCTIONS = [
    compute_euclidean_distance,
    compute_manhattan_distance,
    compute_x2_distance,
    compute_histogram_intersection,
    compute_hellinger_distance,
]
TOPK = [1, 5, 10]  # Values of K for mAP@K evaluation


def compute_keypoint_retrieval(
    bbdd_images,
    query_images,
    gt,
    kp_method,
    desc_method,
    matcher_method,
    all_results,
    force_retrieval: bool = True,
    save_results: bool = True,
    use_ratio_test: bool = True,
    ratio_threshold: float = 0.75,
    visualize: bool = False,
    is_testset: bool = False,
) -> dict:
    kp_name, kp_func = kp_method
    config_name = f"{kp_name}_{desc_method}_{matcher_method}"
    tic = time.perf_counter()
    print(f"\n{'=' * 60}")
    print(f"Testing: {config_name}")
    print(f"{'=' * 60}")

    result_pkl_filename = Path("./df_results/week4_keypoints") / f"{config_name}_results.pkl"

    if result_pkl_filename.exists() and not force_retrieval:
        print(f"Loading cached results from {result_pkl_filename}")
        map_results = pickle.load(open(result_pkl_filename, "rb"))
    else:
        # Perform retrieval using keypoint matching
        try:
            retrieval_results = keypoint_retrieval(
                bbdd_images,
                query_images,
                kp_func,
                descriptor_method=desc_method,
                matcher_method=matcher_method,
                top_k=max(TOPK),
                use_ratio_test=use_ratio_test,
                ratio_threshold=ratio_threshold,
                force_retrieval=force_retrieval,
            )

            # Calculate mAP for different K values
            if not is_testset:
                map_results = {}
                for k in TOPK:
                    # Truncate results to top_k
                    truncated_results = {}
                    for query_idx, crop_results in retrieval_results.items():
                        truncated_results[query_idx] = []
                        for crop_res in crop_results:
                            if crop_res[0][0] < 10:
                                truncated_results[query_idx].append([(0, -1)])
                            else:
                                truncated_results[query_idx].append(crop_res[:k])

                    # Calculate mAP@K
                    map_k = mean_average_precision_K(truncated_results, gt, K=k)
                    map_results[f"mAP@K={k}"] = map_k

                # Save results
                if save_results:
                    pickle.dump(map_results, open(result_pkl_filename, "wb"))
                    # Also save full retrieval results for visualization
                    pickle.dump(
                        retrieval_results,
                        open(result_pkl_filename.with_name(f"{config_name}_full.pkl"), "wb"),
                    )
            else:
                output_results = []
                for i in range(len(retrieval_results)):
                    parcial_results = []
                    for res in retrieval_results[i]:
                        if res[0][0] < 10:
                            res = [(0, -1)]
                        parcial_results.append([r[1] for r in res])
                    output_results.append(parcial_results)
                print(f"Results for {config_name}:")
                print(output_results)
                pickle.dump(
                    output_results,
                    open(
                        f"results_{config_name}_qst1_w4.pkl",
                        "wb",
                    ),
                )
        except Exception as e:
            print(f"Error testing {config_name}: {e}")
    print(f"Testing took {time.perf_counter() - tic:.2f} seconds.")

    # Print results
    if not is_testset:
        print(f"\nResults for {config_name}:")
        for k, map_value in map_results.items():
            print(f"  {k}: {map_value:.4f}")

        all_results[config_name] = map_results

        # Visualize best matches (optional)
        if visualize:
            visualize_keypoint_matches(
                bbdd_images,
                query_images,
                retrieval_results,
                gt,
                kp_func,
                desc_method,
                config_name,
                num_visualize=15,
            )


def compute_test_retrieval(bbdd_images, query_images, descriptor_function, distance_function, query_set_suffix):
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
        parcial_results = []
        for res in results[i]:
            parcial_results.append([r[1] for r in res])
        output_results.append(parcial_results)
    print(f"Results for {descriptor_function.__name__} and {distance_function.__name__}:")
    print(output_results)
    pickle.dump(
        output_results,
        open(
            f"results_{descriptor_function.__name__}_{distance_function.__name__}_{query_set_suffix}.pkl",
            "wb",
        ),
    )


def compute_retrieval_template(
    bbdd_images: dict[str, list[np.ndarray]],
    query_images: dict[str, list[np.ndarray]],
    descriptor_function,
    gt,
    query_set_suffix: str,
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
                for i, list_res in results.items():
                    if len(list_res) == 1:
                        fig, axes = plt.subplots(1, k + 2, figsize=(15, 3))
                        axes = axes.reshape(1, -1)  # Make it 2D for consistent indexing
                    elif len(list_res) > 1:
                        fig, axes = plt.subplots(len(list_res), k + 2, figsize=(15, 6))
                    else:
                        continue

                    fig.suptitle(f"Query: {query_trans[i]}")
                    for j, res in enumerate(list_res):
                        query_img = cv2.cvtColor(query_images[query_trans[i]][j], cv2.COLOR_BGR2RGB)
                        axes[j, 0].imshow(query_img)
                        axes[j, 0].set_title("Query")
                        axes[j, 0].axis("off")
                        try:
                            gt_img = cv2.cvtColor(bbdd_images[bbdd_trans[gt[i][j]]][0], cv2.COLOR_BGR2RGB)
                            axes[j, 1].imshow(gt_img)
                            axes[j, 1].set_title(f"GT: {bbdd_trans[gt[i][j]]}")
                        except IndexError:
                            gt_img = cv2.cvtColor(bbdd_images[bbdd_trans[gt[i][0]]][0], cv2.COLOR_BGR2RGB)
                            axes[j, 1].imshow(gt_img)
                            axes[j, 1].set_title(f"GT: {bbdd_trans[gt[i][0]]}")
                        axes[j, 1].axis("off")
                        for m in range(k):
                            retrieved_img_id = res[m][1]
                            retrieved_img = cv2.cvtColor(
                                bbdd_images[bbdd_trans[retrieved_img_id]][0],
                                cv2.COLOR_BGR2RGB,
                            )
                            axes[j, m + 2].imshow(retrieved_img)
                            axes[j, m + 2].set_title(f"R{m + 1}: {bbdd_trans[retrieved_img_id]}")
                            axes[j, m + 2].axis("off")
                    plt.tight_layout()
                    plt.show()
            map_results[f"mAP@K={k}"][distance_function.__name__] = mean_average_precision_K(results, gt, K=k)
    return map_results


def main():
    # test_weekn_weekm()
    best_of_each_week()


def best_of_each_week():
    week2_best_bins = [4, 4, 2]
    week2_best_grid = (9, 9)
    week3_best_glcm_grid = (12, 12)
    week3_best_dist = [1, 2, 3]
    week3_best_lvl = 16
    week3_best_ncoefs = 30
    week3_best_grid = (3, 3)
    
    harris_params = {"blockSize": 2, "ksize": 3, "k": 0.04,
                    "threshold": 0.05, "nms_radius": 10}
    
    harris_lap_params = {"blockSize": 5,"ksize": 3,"k": 0.04,
                        "threshold": 0.02,"nms_radius": 6,
                        "scales": [1.6, 3.2, 6.4],"max_kps": 2000}

    combinations = [
        ("harris", harris_corner_detection_func(**harris_params), "ORB", "BF-RATIO"),
        ("harris_laplacian", harris_laplacian_detection_func(**harris_lap_params), "COLOR-SIFT", "BF-RATIO"),
    ]

    # Query development datasets
    qsd1_pathlist = list(Path(Path(__file__).parent / "datasets" / "qsd1_w1").glob("*.jpg"))
    qsd2_pathlist = list(Path(Path(__file__).parent / "datasets" / "qsd2_w2").glob("*.jpg"))
    qsd1_3_pathlist = list(Path(Path(__file__).parent / "datasets" / "qsd1_w3").glob("*.jpg"))
    qsd2_3_pathlist = list(Path(Path(__file__).parent / "datasets" / "qsd2_w3").glob("*.jpg"))
    qsd2_3_no_aug_pathlist = list(Path(Path(__file__).parent / "datasets" / "qsd2_w3" / "non_augmented").glob("*.jpg"))
    qsd2_3_masks_pathlist = list(Path(Path(__file__).parent / "datasets" / "qsd2_w3").glob("*.png"))
    qsd1_4_pathlist = list(Path(Path(__file__).parent / "datasets" / "qsd1_w4").glob("*.jpg"))
    qsd1_4_no_aug_pathlist = list(Path(Path(__file__).parent / "datasets" / "qsd1_w4" / "non_augmented").glob("*.jpg"))
    qsd1_4_masks_pathlist = list(Path(Path(__file__).parent / "datasets" / "qsd1_w4").glob("*.png"))

    # Query test datasets
    qst1_pathlist = list(Path(Path(__file__).parent / "datasets" / "qst1_w4").glob("*.jpg"))

    # Dataset
    bbdd_pathlist = list(Path(Path(__file__).parent / "datasets" / "BBDD").glob("*.jpg"))

    # Ground Truths
    gt_qsd1_w3 = pickle.load(open("./datasets/qsd1_w3/gt_corresps.pkl", "rb"))
    gt_qsd2_w3 = pickle.load(open("./datasets/qsd2_w3/gt_corresps.pkl", "rb"))
    gt_qsd1_w4 = pickle.load(open("./datasets/qsd1_w4/gt_corresps.pkl", "rb"))

    # 1.1) Load all images into a dictionary, key is the filename without extension and value is the image in bgr
    bbdd_images = {img_path.stem: [cv2.imread(str(img_path))] for img_path in bbdd_pathlist}
    qsd1_images = {img_path.stem: [cv2.imread(str(img_path))] for img_path in qsd1_pathlist}
    qsd2_images = {img_path.stem: [cv2.imread(str(img_path))] for img_path in qsd2_pathlist}
    qsd1_3_images = {img_path.stem: [cv2.imread(str(img_path))] for img_path in qsd1_3_pathlist}
    qsd2_3_images = {img_path.stem: [cv2.imread(str(img_path))] for img_path in qsd2_3_pathlist}
    qsd2_3_no_aug_images = {img_path.stem: [cv2.imread(str(img_path))] for img_path in qsd2_3_no_aug_pathlist}
    qsd2_3_gt_masks = {
        img_path.stem: [cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)] for img_path in qsd2_3_masks_pathlist
    }
    qsd1_4_images = {img_path.stem: [cv2.imread(str(img_path))] for img_path in qsd1_4_pathlist}
    qsd1_4_no_aug_images = {img_path.stem: [cv2.imread(str(img_path))] for img_path in qsd1_4_no_aug_pathlist}
    qsd1_4_gt_masks = {
        img_path.stem: [cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)] for img_path in qsd1_4_masks_pathlist
    }

    # Test
    qst1_images = {img_path.stem: [cv2.imread(str(img_path))] for img_path in qst1_pathlist}

    # WEEK 4 best segmentation method evaluation on qsd1_w4
    print("segmenting qsd1_w4...")
    # qsd1_4_images = preprocess_images(qsd1_4_images, do_resize=False)
    precision, recall, F1 = 0, 0, 0
    for name, img in qsd1_4_images.items():
        result = get_mask_and_crops_refined(
            img_list=img,
            use_mask="mask_lab",
            min_area=2000,
            reject_border=True,
            border_margin=2,
            outermost_only=True,
        )
        mask = result["mask"]
        crops = result["crops_img"]
        mask_gt = qsd1_4_gt_masks[name][0]

        qsd1_4_images[name] = crops

        metrics = binary_mask_evaluation(mask, mask_gt)

        precision += metrics["precision"]
        recall += metrics["recall"]
        F1 += metrics["F1"]

    avg_precision = precision / len(qsd1_4_images)
    avg_recall = recall / len(qsd1_4_images)
    avg_F1 = F1 / len(qsd1_4_images)
    print(
        f"WEEK 4 (qsd1_w4): Binary Mask metrics: Precision: {avg_precision}. Recall: {avg_recall}. F1-measure: {avg_F1}."
    )

    print("generating perfect crops for qsd1_w4...")
    qsd1_4_perfect_images = {}
    for name, img in qsd1_4_no_aug_images.items():
        mask_gt = qsd1_4_gt_masks[name][0]
        # Generate crops from ground truth mask
        crops_from_gt = get_crops_from_gt_mask(
            img_list=img,
            mask_gt=mask_gt,
            min_area=1500,
            reject_border=False,
            padding=0,
        )
        qsd1_4_perfect_images[name] = crops_from_gt

    # output_masks_dir = Path("./output_masks")
    # output_masks_dir.mkdir(parents=True, exist_ok=True)
    # # WEEK 3 best segmentation method on qst2_w3, generate masks and save it
    # print("segmenting qst2_w3...")
    # qst2_images = preprocess_images(qst2_images, do_resize=False)
    # for name, img in qst2_images.items():
    #     result = get_mask_and_crops(img_list=img, use_mask="mask_lab", min_area=1500)
    #     mask = result["mask"]
    #     crops = result["crops_img"]
    #     cv2.imwrite(str(output_masks_dir / f"{name}.png"), mask)
    #     qst2_images[name] = crops
    # print("segmentation done.\n")

    # Apply resize & denoise preprocessing to all images
    lists_to_preprocess = [
        bbdd_images,
        qsd1_4_images,
        qsd1_4_perfect_images,
        qst1_images,
    ]
    for i in range(len(lists_to_preprocess)):
        lists_to_preprocess[i] = preprocess_images(lists_to_preprocess[i], do_denoise=False)
    lists_to_preprocess = [
        qsd1_4_images,
        qsd1_4_perfect_images,
        qst1_images,
    ]
    for i in range(len(lists_to_preprocess)):
        lists_to_preprocess[i] = preprocess_images(lists_to_preprocess[i], do_resize=False)

    # WEEK 1 best method execution
    map_results = compute_retrieval_template(bbdd_images, qsd1_4_images, hsv_histogram_concat, gt_qsd1_w4, "qsd1_w4")
    print("WEEK 1: hsv_histogram_concat")
    print(pd.DataFrame(map_results))

    # WEEK 2 best method execution on qsd1_w1
    my_hsv_block_hist_concat = hsv_block_hist_concat_func(bins=week2_best_bins, grid=week2_best_grid)
    map_results = compute_retrieval_template(
        bbdd_images, qsd1_4_images, my_hsv_block_hist_concat, gt_qsd1_w4, "qsd1_w4"
    )
    print(f"WEEK 2: hsv_block_hist_concat bins={week2_best_bins} grid={week2_best_grid}, qsd1_w4")
    print(pd.DataFrame(map_results))

    # WEEK 2 best method execution on qsd2_w2
    # map_results = compute_retrieval_template(
    #     bbdd_images, qsd2_images, my_hsv_block_hist_concat, gt_qsd2_w2, "qsd2_w2"
    # )
    # print(
    #     f"WEEK 2: hsv_block_hist_concat bins={week2_best_bins} grid={week2_best_grid}, qsd2_w2"
    # )
    # print(pd.DataFrame(map_results))

    # WEEK 2 best method execution on qsd1_w3
    # map_results = compute_retrieval_template(
    #     bbdd_images, qsd1_3_images, my_hsv_block_hist_concat, gt_qsd1_w3, "qsd1_w3"
    # )
    # print(f"WEEK 2: hsv_block_hist_concat bins={week2_best_bins} grid={week2_best_grid}, qsd1_w3")
    # print(pd.DataFrame(map_results), "\n")

    # WEEK 3 best method execution on qsd1_w4
    my_glcm_block_descriptor = glcm_block_descriptor_func(
        grid=week3_best_glcm_grid,
        distances=week3_best_dist,
        levels=week3_best_lvl,
    )
    map_results = compute_retrieval_template(
        bbdd_images, qsd1_4_images, my_glcm_block_descriptor, gt_qsd1_w4, "qsd1_w4"
    )
    print(f"WEEK 3: GLCM grid={week3_best_glcm_grid} distances={week3_best_dist} levels={week3_best_lvl}, qsd1_w4")
    print(pd.DataFrame(map_results), "\n")
    my_dct_block_descriptor = dct_block_descriptor_func(
        n_coefs=week3_best_ncoefs,
        grid=week3_best_grid,
        relative_coefs=False,
    )
    map_results = compute_retrieval_template(bbdd_images, qsd1_4_images, my_dct_block_descriptor, gt_qsd1_w4, "qsd1_w4")
    print(f"WEEK 3: DCT grid={week3_best_grid} n_coefs={week3_best_ncoefs}, qsd1_w4")
    print(pd.DataFrame(map_results), "\n")

    all_results = {}
    for kp_name, kp_func, desc_method, matcher_method in combinations:
        compute_keypoint_retrieval(
            bbdd_images,
            qsd1_4_images,
            gt_qsd1_w4,
            (kp_name, kp_func),
            desc_method,
            matcher_method,
            all_results=all_results,
            force_retrieval=True,
            save_results=False,
        )
        compute_keypoint_retrieval(
            bbdd_images,
            qst1_images,
            {},
            (kp_name, kp_func),
            desc_method,
            matcher_method,
            all_results=all_results,
            force_retrieval=True,
            save_results=False,
            is_testset=True,
        )

    # WEEK 3 best method execution on qsd1_w4 perfect crops
    # map_results = compute_retrieval_template(
    #     bbdd_images,
    #     qsd1_4_perfect_images,
    #     my_glcm_block_descriptor,
    #     gt_qsd1_w4,
    #     "qsd1_w4_perfect",
    # )
    # print(
    #     f"WEEK 3: GLCM grid={week3_best_glcm_grid} distances={week3_best_dist} levels={week3_best_lvl}, qsd1_w4_perfect"
    # )
    # print(pd.DataFrame(map_results), "\n")
    # map_results = compute_retrieval_template(
    #     bbdd_images,
    #     qsd1_4_perfect_images,
    #     my_dct_block_descriptor,
    #     gt_qsd1_w4,
    #     "qsd1_w4_perfect",
    # )
    # print(f"WEEK 3: DCT grid={week3_best_grid} n_coefs={week3_best_ncoefs}, qsd1_w4_perfect")
    # print(pd.DataFrame(map_results), "\n")

    # # WEEK 3 generate results for qst1_w3
    # compute_test_retrieval(
    #     bbdd_images,
    #     qst1_images,
    #     my_dct_block_descriptor,
    #     compute_hellinger_distance,
    #     "qst1_w3",
    # )
    # # WEEK 3 generate results for qst2_w3
    # compute_test_retrieval(
    #     bbdd_images,
    #     qst2_images,
    #     my_dct_block_descriptor,
    #     compute_hellinger_distance,
    #     "qst2_w3",
    # )


# main function adapted for week 1&2 comparisons:
# 1) load all images in bgr
# 2) preprocess all images
# 3) compute week 1 method as reference point
# 4) compute week 2 methods with different parameters (nested loops, takes a long time to compute)
def test_weekn_weekm(weekn: int = 4, weekm: int = 4):
    tic_init = time.perf_counter()
    print(f"Testing week {weekn} vs week {weekm} methods...")
    force_retrieval = True  # If True, forces recomputation of descriptors and retrieval even if result pkl files exist
    save_results = True  # If True, saves results of retrieval in method_bins_grids.pkl
    test_mode = False  # If True, only process a few images for quick testing & visualization purposes
    harris_params = {"blockSize": 3, "ksize": 3, "k": 0.06, "threshold": 0.001, "nms_radius": 3}
    harris_lap_params = {
        "blockSize": 5,
        "ksize": 3,
        "k": 0.04,
        "threshold": 0.02,
        "nms_radius": 6,
        "scales": [1.6, 3.2, 6.4],
        "max_kps": 2000,
    }

    combinations = [
        # ("harris", harris_corner_detection_func(**harris_params), "ORB", "BF-RATIO"),
        # ("harris", harris_corner_detection_func(**harris_params), "SIFT", "BF-RATIO"),
        # ("harris", harris_corner_detection_func(**harris_params), "COLOR-SIFT", "BF-RATIO"),
        # ("dog", dog_detection, "ORB", "BF-RATIO"),
        # ("dog", dog_detection, "SIFT", "BF-RATIO"),
        # ("dog", dog_detection, "COLOR-SIFT", "BF-RATIO"),
        # ("harris_laplacian", harris_laplacian_detection_func(**harris_lap_params), "ORB", "BF-RATIO"),
        # ("harris_laplacian", harris_laplacian_detection_func(**harris_lap_params), "SIFT", "BF-RATIO"),
        ("harris_laplacian", harris_laplacian_detection_func(**harris_lap_params), "COLOR-SIFT", "BF-RATIO"),
    ]

    # Create results directory if it doesn't exist
    if save_results:
        Path("./df_results/week4_keypoints").mkdir(parents=True, exist_ok=True)

    # 1) Load all images paths
    if test_mode:
        qsd1_pathlist = list(Path(Path(__file__).parent / "datasets" / "qsd1_w1").glob("*.jpg"))[:5]
        qsd1_w4_pathlist = list(Path(Path(__file__).parent / "datasets" / "qsd1_w4").glob("*.jpg"))[:5]
    else:
        qsd1_w4_pathlist = list(Path(Path(__file__).parent / "datasets" / "qsd1_w4").glob("*.jpg"))
        qsd1_w4_gt_masks_pathlist = list(Path(Path(__file__).parent / "datasets" / "qsd1_w4").glob("*.png"))
    bbdd_pathlist = list(Path(Path(__file__).parent / "datasets" / "BBDD").glob("*.jpg"))

    # Load ground truth correspondences
    gt_qsd1_w4 = pickle.load(open("./datasets/qsd1_w4/gt_corresps.pkl", "rb"))

    # 1.1) Load all images into a dictionary, key is the filename without extension and value is the image in bgr
    bbdd_images = {img_path.stem: [cv2.imread(str(img_path))] for img_path in bbdd_pathlist}
    qsd1_w4_images = {img_path.stem: [cv2.imread(str(img_path))] for img_path in qsd1_w4_pathlist}
    qsd1_w4_gt_masks = {
        img_path.stem: [cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)] for img_path in qsd1_w4_gt_masks_pathlist
    }

    # Segment query images to get crops
    print(f"Loading images took {time.perf_counter() - tic_init:.2f} seconds.")
    print("Segmenting qsd1_w4 using our method...")
    tic = time.perf_counter()
    qsd1_4_crops = {}
    for name, img in qsd1_w4_images.items():
        result = get_mask_and_crops_refined(
            img_list=img,
            use_mask="mask_lab",
            min_area=2000,
            reject_border=True,
            border_margin=2,
            outermost_only=True,
        )
        qsd1_4_crops[name] = result["crops_img"]
    # print("Segmenting qsd1_w4 using gt masks...")
    # tic = time.perf_counter()
    # qsd1_w4_gt_crops = {}
    # for name, img in qsd1_w4_images.items():
    #     mask_gt = qsd1_w4_gt_masks[name][0]
    #     # Generate crops from ground truth mask
    #     crops_from_gt = get_crops_from_gt_mask(
    #         img_list=img,
    #         mask_gt=mask_gt,
    #         min_area=1500,
    #         reject_border=False,
    #         padding=0,
    #     )
    #     qsd1_w4_gt_crops[name] = crops_from_gt
    print(f"Segmenting block took {time.perf_counter() - tic:.2f} seconds.")

    # 2) Preprocess all images with the same preprocessing method (resize 256x256, color balance, contrast&brightness adjustment, smoothing)
    print(f"Preprocessing images... (from init took {time.perf_counter() - tic_init:.2f} seconds)")
    tic = time.perf_counter()
    lists_to_preprocess = [bbdd_images, qsd1_4_crops]
    for i in range(len(lists_to_preprocess)):
        preprocess_images(lists_to_preprocess[i], do_denoise=False)

    preprocess_images(qsd1_4_crops, do_resize=False)

    # lists_to_preprocess = [qsd1_4_crops]
    # for i in range(len(lists_to_preprocess)):
    #     preprocess_images_2(lists_to_preprocess[i], do_denoise=True)

    all_results = {}
    print(f"Preprocessing took {time.perf_counter() - tic:.2f} seconds.")

    print(f"Starting keypoint methods testing... (from init took {time.perf_counter() - tic_init:.2f} seconds)")
    # Testing loop
    tic = time.perf_counter()
    all_results = {}
    for kp_name, kp_func, desc_method, matcher_method in combinations:
        try:
            compute_keypoint_retrieval(
                bbdd_images,
                qsd1_4_crops,
                gt_qsd1_w4,
                (kp_name, kp_func),
                desc_method,
                matcher_method,
                all_results,
                force_retrieval,
                save_results,
            )
        except Exception as e:
            print(f"Error testing {kp_name}_{desc_method}_{matcher_method}: {e}")
    print(f"All keypoint methods tested in {time.perf_counter() - tic:.2f} seconds.")
    # Create summary DataFrame
    summary_df = pd.DataFrame(all_results).T
    print("\n" + "=" * 60)
    print("SUMMARY OF ALL KEYPOINT METHODS")
    print("=" * 60)
    print(summary_df)

    if save_results:
        summary_df.to_csv("./df_results/week4_keypoints/summary.csv")
        print("\nSummary saved to ./df_results/week4_keypoints/summary.csv")


def visualize_keypoint_matches(
    bbdd_images,
    query_images,
    retrieval_results,
    gt,
    keypoint_func,
    descriptor_method,
    config_name,
    num_visualize=3,
):
    """
    Visualize keypoint matches for the best retrievals.
    """
    bbdd_trans = {int(k.split("_")[-1]): k for k in bbdd_images.keys()}
    query_trans = {int(k): k for k in query_images.keys()}

    visualized = 0
    for query_idx, crop_results in retrieval_results.items():
        if visualized >= num_visualize:
            break
        query_name = query_trans[query_idx]
        query_crops = query_images[query_name]

        for crop_idx, results in enumerate(crop_results):
            if visualized >= num_visualize:
                break
            # Get best match
            best_match_n, best_match_idx = results[0]
            if best_match_idx != -1:
                bbdd_name = bbdd_trans[best_match_idx]
            else:
                bbdd_name = "N/A"

            # Get GT match
            try:
                gt_idx = gt[query_idx][crop_idx]
                gt_name = bbdd_trans[gt_idx]
                is_correct = best_match_idx == gt_idx
            except (IndexError, KeyError):
                gt_name = "N/A"
                is_correct = False
                if best_match_n < 10 and gt_idx == -1:
                    is_correct = True

            # Compute keypoints and descriptors
            query_img = query_crops[crop_idx]
            if bbdd_name == "N/A":
                bbdd_img = query_img
            else:
                bbdd_img = bbdd_images[bbdd_name][0]

            kps_query = keypoint_func(query_img.copy())
            kps_bbdd = keypoint_func(bbdd_img.copy())

            desc_query = compute_local_descriptors(query_img, kps_query, method=descriptor_method)
            desc_bbdd = compute_local_descriptors(bbdd_img, kps_bbdd, method=descriptor_method)

            n_matches, matches = calculate_matches(desc_query, desc_bbdd)

            # Draw matches
            match_img = cv2.drawMatches(
                query_img,
                kps_query,
                bbdd_img,
                kps_bbdd,
                matches[:80],  # Show top 80 matches
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )

            if best_match_n < 10:
                bbdd_name = -1

            savepath = Path("./visualize") / "week4_keypoint_matches"
            savepath.mkdir(parents=True, exist_ok=True)
            # Display
            plt.figure(figsize=(15, 5))
            plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
            title = f"{config_name}\n"
            title += f"Query: {query_name} (crop {crop_idx}) -> Retrieved: {bbdd_name}\n"
            title += f"Matches: {n_matches} | GT: {gt_name} | "
            title += f"{'CORRECT' if is_correct else 'INCORRECT'}"
            plt.title(title)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(savepath / f"{config_name}_query{query_name}_crop{crop_idx}.png")
            plt.close()
            visualized += 1


if __name__ == "__main__":
    main()
