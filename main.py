from pathlib import Path
import pickle
import cv2

from descriptors import (
    compute_descriptors,
    hsv_histogram_concat,
    hsv_hier_block_hist_concat_func,
    hsv_block_hist_concat_func
)
from similarity import (
    compute_euclidean_distance,
    compute_manhattan_distance,
    compute_x2_distance,
    compute_histogram_intersection,
    compute_hellinger_distance,
)
from retrieval import retrieval
from metrics import mean_average_precision_K, BinaryMaskEvaluation
from preprocess import preprocess_images
import pandas as pd

# main function adapted for week 1&2 comparisons:
# 1) load all images in bgr
# 2) preprocess all images
# 3) compute week 1 method as reference point
# 4) compute week 2 methods with different parameters (nested loops, takes a long time to compute)
def main():
    test_mode = False  # If True, only process a few images for quick testing & visualization purposes
    TOPK = [1, 5, 10]  # Values of K for mAP@K evaluation
    distance_functions = [
        compute_euclidean_distance,
        compute_manhattan_distance,
        compute_x2_distance,
        compute_histogram_intersection,
        compute_hellinger_distance
    ]
    testing_bins = [[8,8,8], [16,16,16], [32,32,32], [64,64,64]]
    testing_grids = [(1,1), (2,2), (3,3), (4,4), (5,5)]
    testing_level_grids = [[(1,1), (2,2), (3,3)], [(2,2), (3,3), (4,4)], [(3,3), (4,4), (5,5)], [(1,1), (2,2), (4,4)], [(1,1), (3,3), (5,5)]]
    
    # 1) Load all images paths
    if test_mode:
        qsd1_pathlist = list(Path(Path(__file__).parent / "datasets" / "qsd1_w1").glob("*.jpg"))[:5]
        qsd2_pathlist = list(Path(Path(__file__).parent / "datasets" / "qsd2_w2").glob("*.jpg"))[:5]
        qsd2_masks_pathlist = list(Path(Path(__file__).parent / "datasets" / "qsd2_w2").glob("*.png"))[:5]
        qst1_pathlist = list(Path(Path(__file__).parent / "datasets" / "qst1_w1").glob("*.jpg"))[:5]
        qst2_pathlist = list(Path(Path(__file__).parent / "datasets" / "qst2_w2").glob("*.jpg"))[:5]
    else:
        qsd1_pathlist = list(Path(Path(__file__).parent / "datasets" / "qsd1_w1").glob("*.jpg"))
        qsd2_pathlist = list(Path(Path(__file__).parent / "datasets" / "qsd2_w2").glob("*.jpg"))
        qsd2_masks_pathlist = list(Path(Path(__file__).parent / "datasets" / "qsd2_w2").glob("*.png"))
        qst1_pathlist = list(Path(Path(__file__).parent / "datasets" / "qst1_w1").glob("*.jpg"))
        qst2_pathlist = list(Path(Path(__file__).parent / "datasets" / "qst2_w2").glob("*.jpg"))
    bbdd_pathlist = list(Path(Path(__file__).parent / "datasets" / "BBDD").glob("*.jpg"))
    
    # Load qsd1_w1 ground truth correspondences
    gt = pickle.load(open("./datasets/qsd1_w1/gt_corresps.pkl", "rb"))

    # 1.1) Load all images into a dictionary, key is the filename without extension and value is the image in bgr
    bbdd_images = {img_path.stem: cv2.imread(str(img_path)) for img_path in bbdd_pathlist}
    qsd1_images = {img_path.stem: cv2.imread(str(img_path)) for img_path in qsd1_pathlist}
    qsd2_images = {img_path.stem: cv2.imread(str(img_path)) for img_path in qsd2_pathlist}
    qst1_images = {img_path.stem: cv2.imread(str(img_path)) for img_path in qst1_pathlist}
    qst2_images = {img_path.stem: cv2.imread(str(img_path)) for img_path in qst2_pathlist}
    qsd2_gt_masks = {img_path.stem: cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE) for img_path in qsd2_masks_pathlist}

    """
    TODO WEEK 2 TASK 4:
        - Compute binary masks of query_images
        - loop through query_gt_masks and the computed binary masks calling metrics.BinaryMaskEvaluation
        possible pseudocode:
            average_metrics = {}
            binary_masks, cropped_images = segment_background(qsd2_images)
            for img_name in qsd2_gt_masks.keys():
                result = BinaryMaskEvaluation(binary_masks[img_name], qsd2_gt_masks[img_name])
                for key in average_metrics.keys():
                    if key not in average_metrics:
                        average_metrics[key] = 0.0
                    average_metrics[key] += result[key]
            for key in average_metrics.keys():
                average_metrics[key] /= len(qsd2_gt_masks)
            print(average_metrics)
    """
    
    # 2) Preprocess all images with the same preprocessing method (resize 256x256, color balance, contrast&brightness adjustment, smoothing)
    lists_to_preprocess = [bbdd_images, qsd1_images, qst1_images]
    for i in range(len(lists_to_preprocess)):
        lists_to_preprocess[i] = preprocess_images(lists_to_preprocess[i])

    # WEEK 1 method 1: hsv_histogram_concat x all distance metrics
    bbdd_rgb_descriptors = compute_descriptors(
        "bbdd",
        hsv_histogram_concat,
        bbdd_images,
        use_grayscale=False,
        save_as_pkl=True,
        overwrite_pkl=True,
    )

    # Repeate the same for Query descriptors of qsd1_w1
    query_rgb_descriptors = compute_descriptors(
        "qsd1_w1",
        hsv_histogram_concat,
        qsd1_images,
        use_grayscale=False,
        save_as_pkl=True,
        overwrite_pkl=True,
    )
    
    map_results = {}
    for k in TOPK:
        map_results[f"mAP@K={k}"] = {}
        for distance_function in distance_functions:
            results = retrieval(
                bbdd_rgb_descriptors,
                query_rgb_descriptors,
                distance_function,
                top_k=k,
            )
            map_results[f"mAP@K={k}"][distance_function.__name__] = mean_average_precision_K(results, gt, K=k)
    print("WEEK 1: hsv_histogram_concat")
    print(pd.DataFrame(map_results))
    print()

    # WEEK 2 methods 1 & 2: hsv_block_hist_concat & hsv_hier_block_hist_concat x all distance metrics
    for bins in testing_bins:
        for grid_idx, grid in enumerate(testing_grids):
            # START method hsv_block_hist_concat
            my_hsv_block_hist_concat = hsv_block_hist_concat_func(bins=bins, grid=grid)
            bbdd_rgb_descriptors = compute_descriptors(
                "bbdd",
                my_hsv_block_hist_concat,
                bbdd_images,
                use_grayscale=False,
                save_as_pkl=True,
                overwrite_pkl=True,
            )

            # Repeate the same for Query descriptors of qsd1_w1
            query_rgb_descriptors = compute_descriptors(
                "qsd1_w1",
                my_hsv_block_hist_concat,
                qsd1_images,
                use_grayscale=False,
                save_as_pkl=True,
                overwrite_pkl=True,
            )
            
            map_results = {}
            for k in TOPK:
                map_results[f"mAP@K={k}"] = {}
                for distance_function in distance_functions:
                    results = retrieval(
                        bbdd_rgb_descriptors,
                        query_rgb_descriptors,
                        distance_function,
                        top_k=k,
                    )
                    map_results[f"mAP@K={k}"][distance_function.__name__] = mean_average_precision_K(results, gt, K=k)
            print(f"WEEK 2: hsv_block_hist_concat bins={bins} grid={grid}")
            print(pd.DataFrame(map_results))
            print()
            # END method hsv_block_hist_concat
            
            # START method hsv_hier_block_hist_concat
            my_hsv_hier_block_hist_concat = hsv_hier_block_hist_concat_func(bins=bins, levels_grid=testing_level_grids[grid_idx])
            bbdd_rgb_descriptors = compute_descriptors(
                "bbdd",
                my_hsv_hier_block_hist_concat,
                bbdd_images,
                use_grayscale=False,
                save_as_pkl=True,
                overwrite_pkl=True,
            )

            # Repeate the same for Query descriptors of qsd1_w1
            query_rgb_descriptors = compute_descriptors(
                "qsd1_w1",
                my_hsv_hier_block_hist_concat,
                qsd1_images,
                use_grayscale=False,
                save_as_pkl=True,
                overwrite_pkl=True,
            )
            
            map_results = {}
            for k in TOPK:
                map_results[f"mAP@K={k}"] = {}
                for distance_function in distance_functions:
                    results = retrieval(
                        bbdd_rgb_descriptors,
                        query_rgb_descriptors,
                        distance_function,
                        top_k=k,
                    )
                    map_results[f"mAP@K={k}"][distance_function.__name__] = mean_average_precision_K(results, gt, K=k)
            print(f"WEEK 2: hsv_hier_block_hist_concat bins={bins} levels_grid={testing_level_grids[grid_idx]}")
            print(pd.DataFrame(map_results))
            print()
            # END method hsv_hier_block_hist_concat

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
