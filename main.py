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
from segmentation import compute_binary_mask_1, compute_binary_mask_2
from retrieval import retrieval
from preprocess import preprocess_images, preprocess_images_for_segmentation
from metrics import mean_average_precision_K, binary_mask_evaluation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_retrieval_template(
    bbdd_images,
    query_images,
    descriptor_function,
    distance_func_list,
    topk_list,
    gt,
    query_set_suffix
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
    for k in topk_list:
        map_results[f"mAP@K={k}"] = {}
        for distance_function in distance_func_list:
            results = retrieval(
                bbdd_descriptors,
                query_descriptors,
                distance_function,
                top_k=k,
            )
            map_results[f"mAP@K={k}"][distance_function.__name__] = mean_average_precision_K(results, gt, K=k)
    return map_results

# main function adapted for week 1&2 comparisons:
# 1) load all images in bgr
# 2) preprocess all images
# 3) compute week 1 method as reference point
# 4) compute week 2 methods with different parameters (nested loops, takes a long time to compute)
def main():
    run_block_histogram_concat = True
    run_hier_block_histogram_concat = True
    force_retrieval = False # If True, forces recomputation of descriptors and retrieval even if result pkl files exist
    save_results = True  # If True, saves results of retrieval in method_bins_grids.pkl
    test_mode = False  # If True, only process a few images for quick testing & visualization purposes
    TOPK = [1, 5, 10]  # Values of K for mAP@K evaluation
    distance_functions = [
        compute_euclidean_distance,
        compute_manhattan_distance,
        compute_x2_distance,
        compute_histogram_intersection,
        compute_hellinger_distance
    ]
    testing_bins = [[4,4,2], [6,6,3], [8,8,4], [10,10,5], [12,12,6], [14,14,7], [16,16,8], [32,32,16], [64,64,32], [128,128,64]]
    testing_grids = [(1,1), (2,2), (3,3), (4,4), (5,5), (6,6), (7,7), (8,8), (9,9), (10,10), (11,11), (12,12)]
    testing_level_grids = [[(1,1), (2,2)], [(2,2), (4,4)], [(4,4), (8,8)], [(3,3), (9,9)]]
    
    # Create results directory if it doesn't exist
    if save_results:
        Path("./df_results").mkdir(parents=True, exist_ok=True)
    
    # 1) Load all images paths
    if test_mode:
        qsd1_pathlist = list(Path(Path(__file__).parent / "datasets" / "qsd1_w1").glob("*.jpg"))[:5]
        qsd2_pathlist = list(Path(Path(__file__).parent / "datasets" / "qsd2_w1").glob("*.jpg"))[:5]
        qsd2_masks_pathlist = list(Path(Path(__file__).parent / "datasets" / "qsd2_w1").glob("*.png"))[:5]
        qst1_pathlist = list(Path(Path(__file__).parent / "datasets" / "qst1_w1").glob("*.jpg"))[:5]
        qst2_pathlist = list(Path(Path(__file__).parent / "datasets" / "qst2_w2").glob("*.jpg"))[:5]
    else:
        qsd1_pathlist = list(Path(Path(__file__).parent / "datasets" / "qsd1_w1").glob("*.jpg"))
        qsd2_pathlist = list(Path(Path(__file__).parent / "datasets" / "qsd2_w1").glob("*.jpg"))
        qsd2_masks_pathlist = list(Path(Path(__file__).parent / "datasets" / "qsd2_w1").glob("*.png"))
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
    
    precision, recall, F1 = 0, 0, 0
    qsd2_masked_images = {}
    for name, img in qsd2_images.items():
        
        mask = compute_binary_mask_1(img)
        mask_gt = qsd2_gt_masks[name]
        
        plt.imshow(img)
        plt.show()
        plt.imshow(mask, cmap="grey")
        plt.show()
        
        mask_bool = mask > 0
        masked_rgb = img * mask_bool[:, :, np.newaxis]
        qsd2_masked_images[name] = masked_rgb
        
        plt.imshow(masked_rgb)
        plt.show()
        
        metrics = binary_mask_evaluation(mask, mask_gt)
        
        precision += metrics['precision']
        recall += metrics['recall']
        F1 += metrics['F1']
    
    avg_precision = precision / len(qsd2_images)
    avg_recall = recall / len(qsd2_images)
    avg_F1 = F1 / len(qsd2_images)
    print(f"WEEK 2: Binary Mask metrics: Precision: {avg_precision}. Recall: {avg_recall}. F1-measure: {avg_F1}.")
            
    # 2) Preprocess all images with the same preprocessing method (resize 256x256, color balance, contrast&brightness adjustment, smoothing)
    lists_to_preprocess = [bbdd_images, qsd1_images, qst1_images]
    for i in range(len(lists_to_preprocess)):
        lists_to_preprocess[i] = preprocess_images(lists_to_preprocess[i])

    result_pkl_filename = Path("./df_results") / "results_hsv_histogram_concat.pkl"
    if result_pkl_filename.exists() and not force_retrieval:
        map_results = pickle.load(open(result_pkl_filename, "rb"))
        print("WEEK 1: hsv_histogram_concat")
        print(pd.DataFrame(map_results))
    else:
        # WEEK 1 method 1: hsv_histogram_concat x all distance metrics
        map_results = compute_retrieval_template(
            bbdd_images,
            qsd1_images,
            hsv_histogram_concat,
            distance_functions,
            TOPK,
            gt,
            "qsd1_w1"
        )
        print("WEEK 1: hsv_histogram_concat")
        print(pd.DataFrame(map_results))
        print()
        if save_results:
            pickle.dump(
                map_results, open(result_pkl_filename, "wb")
            )

    # WEEK 2 methods 1 & 2: hsv_block_hist_concat & hsv_hier_block_hist_concat x all distance metrics
    for bins in testing_bins:
        for grid_idx, grid in enumerate(testing_grids):
            if run_block_histogram_concat:
                # START method hsv_block_hist_concat
                result_pkl_filename = Path("./df_results") / f"results_hsv_block_hist_concat_bins_{bins[0]}-{bins[1]}-{bins[2]}_grid_{grid[0]}-{grid[1]}.pkl"
                if result_pkl_filename.exists() and not force_retrieval:
                    map_results = pickle.load(open(result_pkl_filename, "rb"))
                    print(f"WEEK 2: hsv_block_hist_concat bins={bins} grid={grid}")
                    print(pd.DataFrame(map_results))
                    continue
                my_hsv_block_hist_concat = hsv_block_hist_concat_func(bins=bins, grid=grid)
                map_results = compute_retrieval_template(
                    bbdd_images,
                    qsd1_images,
                    my_hsv_block_hist_concat,
                    distance_functions,
                    TOPK,
                    gt,
                    "qsd1_w1"
                )
                print(f"WEEK 2: hsv_block_hist_concat bins={bins} grid={grid}")
                print(pd.DataFrame(map_results))
                print()
                if save_results:
                    pickle.dump(
                        map_results, open(result_pkl_filename, "wb")
                    )
                # END method hsv_block_hist_concat

            if run_hier_block_histogram_concat and grid_idx < len(testing_level_grids):
                # START method hsv_hier_block_hist_concat
                result_pkl_filename = Path("./df_results") / f"results_hsv_hier_block_hist_concat_bins_{bins[0]}-{bins[1]}-{bins[2]}_levels_grid_{testing_level_grids[grid_idx]}.pkl"
                if result_pkl_filename.exists() and not force_retrieval:
                    map_results = pickle.load(open(result_pkl_filename, "rb"))
                    print(f"WEEK 2: hsv_hier_block_hist_concat bins={bins} levels_grid={testing_level_grids[grid_idx]}")
                    print(pd.DataFrame(map_results))
                    continue
                my_hsv_hier_block_hist_concat = hsv_hier_block_hist_concat_func(bins=bins, levels_grid=testing_level_grids[grid_idx])
                map_results = compute_retrieval_template(
                    bbdd_images,
                    qsd1_images,
                    my_hsv_hier_block_hist_concat,
                    distance_functions,
                    TOPK,
                    gt,
                    "qsd1_w1"
                )
                print(f"WEEK 2: hsv_hier_block_hist_concat bins={bins} levels_grid={testing_level_grids[grid_idx]}")
                print(pd.DataFrame(map_results))
                print()
                if save_results:
                    pickle.dump(
                        map_results, open(result_pkl_filename, "wb")
                    )
                # END method hsv_hier_block_hist_concat
                
    # WEEK 2: Segmentation


if __name__ == "__main__":
    main()
