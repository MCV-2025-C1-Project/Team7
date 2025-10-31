from pathlib import Path
import pickle
import cv2
from descriptors import (
    compute_descriptors,
    hsv_block_hist_concat_func,
    lbp_descriptor_histogram_func,
    dct_block_descriptor_func,
    glcm_block_descriptor_func,
)
from similarity import (
    compute_euclidean_distance,
    compute_manhattan_distance,
    compute_x2_distance,
    compute_histogram_intersection,
    compute_hellinger_distance,
)
from keypoints import (
    harris_corner_detection,
    harris_laplacian_detection,
    dog_detection,
    to_keypoints,
    compute_local_descriptors
)
from segmentation import get_mask_and_crops, get_crops_from_gt_mask
from retrieval import retrieval
from preprocess import preprocess_images
from metrics import mean_average_precision_K, binary_mask_evaluation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from skimage.metrics import structural_similarity as ssim_metric

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
        parcial_results = []
        for res in results[i]:
            parcial_results.append([r[1] for r in res])
        output_results.append(parcial_results)
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
                        query_img = cv2.cvtColor(
                            query_images[query_trans[i]][j], cv2.COLOR_BGR2RGB
                        )
                        axes[j, 0].imshow(query_img)
                        axes[j, 0].set_title("Query")
                        axes[j, 0].axis("off")
                        try:
                            gt_img = cv2.cvtColor(
                                bbdd_images[bbdd_trans[gt[i][j]]][0], cv2.COLOR_BGR2RGB
                            )
                            axes[j, 1].imshow(gt_img)
                            axes[j, 1].set_title(f"GT: {bbdd_trans[gt[i][j]]}")
                        except IndexError:
                            gt_img = cv2.cvtColor(
                                bbdd_images[bbdd_trans[gt[i][0]]][0], cv2.COLOR_BGR2RGB
                            )
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
                            axes[j, m + 2].set_title(
                                f"R{m + 1}: {bbdd_trans[retrieved_img_id]}"
                            )
                            axes[j, m + 2].axis("off")
                    plt.tight_layout()
                    plt.show()
            map_results[f"mAP@K={k}"][distance_function.__name__] = (
                mean_average_precision_K(results, gt, K=k)
            )
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

    # Query development datasets
    qsd1_pathlist = list(
        Path(Path(__file__).parent / "datasets" / "qsd1_w1").glob("*.jpg")
    )
    qsd2_pathlist = list(
        Path(Path(__file__).parent / "datasets" / "qsd2_w2").glob("*.jpg")
    )
    qsd1_3_pathlist = list(
        Path(Path(__file__).parent / "datasets" / "qsd1_w3").glob("*.jpg")
    )
    qsd2_3_pathlist = list(
        Path(Path(__file__).parent / "datasets" / "qsd2_w3").glob("*.jpg")
    )
    qsd2_3_no_aug_pathlist = list(
        Path(Path(__file__).parent / "datasets" / "qsd2_w3" / "non_augmented").glob("*.jpg")
    )
    qsd2_3_masks_pathlist = list(
        Path(Path(__file__).parent / "datasets" / "qsd2_w3").glob("*.png")
    )
    qsd1_4_pathlist = list(
        Path(Path(__file__).parent / "datasets" / "qsd1_w4").glob("*.jpg")
    )
    qsd1_4_no_aug_pathlist = list(
        Path(Path(__file__).parent / "datasets" / "qsd1_w4" / "non_augmented").glob("*.jpg")
    )
    qsd1_4_masks_pathlist = list(
        Path(Path(__file__).parent / "datasets" / "qsd1_w4").glob("*.png")
    )
    
    # Query test datasets
    qst1_pathlist = list(
        Path(Path(__file__).parent / "datasets" / "qst1_w3").glob("*.jpg")
    )
    qst2_pathlist = list(
        Path(Path(__file__).parent / "datasets" / "qst2_w3").glob("*.jpg")
    )
    
    # Dataset
    bbdd_pathlist = list(
        Path(Path(__file__).parent / "datasets" / "BBDD").glob("*.jpg")
    )
    
    # Ground Truths
    gt_qsd1_w3 = pickle.load(open("./datasets/qsd1_w3/gt_corresps.pkl", "rb"))
    gt_qsd2_w3 = pickle.load(open("./datasets/qsd2_w3/gt_corresps.pkl", "rb"))
    gt_qsd1_w4 = pickle.load(open("./datasets/qsd1_w4/gt_corresps.pkl", "rb"))

    # 1.1) Load all images into a dictionary, key is the filename without extension and value is the image in bgr
    bbdd_images = {
        img_path.stem: [cv2.imread(str(img_path))] for img_path in bbdd_pathlist
    }
    qsd1_images = {
        img_path.stem: [cv2.imread(str(img_path))] for img_path in qsd1_pathlist
    }
    qsd2_images = {
        img_path.stem: [cv2.imread(str(img_path))] for img_path in qsd2_pathlist
    }
    qsd1_3_images = {
        img_path.stem: [cv2.imread(str(img_path))] for img_path in qsd1_3_pathlist
    }
    qsd2_3_images = {
        img_path.stem: [cv2.imread(str(img_path))] for img_path in qsd2_3_pathlist
    }
    qsd2_3_no_aug_images = {
        img_path.stem: [cv2.imread(str(img_path))]
        for img_path in qsd2_3_no_aug_pathlist
    }
    qsd2_3_gt_masks = {
        img_path.stem: [cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)]
        for img_path in qsd2_3_masks_pathlist
    }
    qsd1_4_images = {
        img_path.stem: [cv2.imread(str(img_path))] for img_path in qsd1_4_pathlist
    }
    qsd1_4_no_aug_images = {
        img_path.stem: [cv2.imread(str(img_path))]
        for img_path in qsd1_4_no_aug_pathlist
    }
    qsd1_4_gt_masks = {
        img_path.stem: [cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)]
        for img_path in qsd1_4_masks_pathlist
    }
    
    # Test
    qst1_images = {
        img_path.stem: [cv2.imread(str(img_path))] for img_path in qst1_pathlist
    }
    qst2_images = {
        img_path.stem: [cv2.imread(str(img_path))] for img_path in qst2_pathlist
    }
    
    # WEEK 4 keypoint detection and local descriptor computation
    for name, img in qsd1_4_images.items():
        harris = copy.deepcopy(img[0])
        harris_lap = copy.deepcopy(img[0])
        dog = copy.deepcopy(img[0])
        
        points_harris = harris_corner_detection(harris)
        kps_harris = to_keypoints(points_harris, size=3)
        points_harris_lap = harris_laplacian_detection(harris_lap)
        kps_harris_lap = to_keypoints(harris_laplacian_detection(harris_lap), size=3)
        kps_dog = dog_detection(dog)
        
        kps, desc = compute_local_descriptors(img[0], kps_harris, method='SIFT')

        print(f"{len(kps)} keypoints, descriptor shape = {desc.shape}")
        
        # Visualize
        img_out = cv2.drawKeypoints(img[0], kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
        plt.title("Harris + SIFT descriptors")
        plt.axis("off")
        plt.show()
        
        """
        methods = ['SIFT', 'ORB', 'AKAZE']

        """
    # WEEK 3 best segmentation method evaluation on qsd2_w3
    print("segmenting qsd2_w3...")
    qsd2_3_images = preprocess_images(qsd2_3_images, do_resize=False)
    precision, recall, F1 = 0, 0, 0
    for name, img in qsd2_3_images.items():
        result = get_mask_and_crops(img_list=img, use_mask="mask_lab", min_area=1500)
        mask = result["mask"]
        crops = result["crops_img"]
        mask_gt = qsd2_3_gt_masks[name][0]

        qsd2_3_images[name] = crops

        metrics = binary_mask_evaluation(mask, mask_gt)

        precision += metrics["precision"]
        recall += metrics["recall"]
        F1 += metrics["F1"]

    avg_precision = precision / len(qsd2_3_images)
    avg_recall = recall / len(qsd2_3_images)
    avg_F1 = F1 / len(qsd2_3_images)
    print(
        f"WEEK 3 (qsd2_w3): Binary Mask metrics: Precision: {avg_precision}. Recall: {avg_recall}. F1-measure: {avg_F1}."
    )

    print("generating perfect crops for qsd2_w3...")
    qsd2_3_perfect_images = {}
    for name, img in qsd2_3_no_aug_images.items():
        mask_gt = qsd2_3_gt_masks[name][0]
        # Generate crops from ground truth mask
        crops_from_gt = get_crops_from_gt_mask(
            img_bgr=img[0],
            mask_gt=mask_gt,
            min_area=1500,
            reject_border=False,
            padding=0,
        )
        qsd2_3_perfect_images[name] = crops_from_gt

    output_masks_dir = Path("./output_masks")
    output_masks_dir.mkdir(parents=True, exist_ok=True)
    # WEEK 3 best segmentation method on qst2_w3, generate masks and save it
    print("segmenting qst2_w3...")
    qst2_images = preprocess_images(qst2_images, do_resize=False)
    for name, img in qst2_images.items():
        result = get_mask_and_crops(img_list=img, use_mask="mask_lab", min_area=1500)
        mask = result["mask"]
        crops = result["crops_img"]
        cv2.imwrite(str(output_masks_dir / f"{name}.png"), mask)
        qst2_images[name] = crops
    print("segmentation done.\n")

    # Apply resize & denoise preprocessing to all images
    lists_to_preprocess = [
        qsd1_images,
        qsd2_images,
        qst1_images,
    ]
    for i in range(len(lists_to_preprocess)):
        lists_to_preprocess[i] = preprocess_images(lists_to_preprocess[i])

    lists_to_preprocess = [
        bbdd_images,
        qsd1_3_images,
        qsd2_3_images,
        qst2_images,
        qsd2_3_perfect_images,
    ]
    for i in range(len(lists_to_preprocess)):
        lists_to_preprocess[i] = preprocess_images(
            lists_to_preprocess[i], do_denoise=False
        )

    # WEEK 1 best method execution
    # map_results = compute_retrieval_template(
    #     bbdd_images, qsd1_images, hsv_histogram_concat, gt_qsd1_w1, "qsd1_w1"
    # )
    # print("WEEK 1: hsv_histogram_concat")
    # print(pd.DataFrame(map_results))

    # WEEK 2 best method execution on qsd1_w1
    my_hsv_block_hist_concat = hsv_block_hist_concat_func(
        bins=week2_best_bins, grid=week2_best_grid
    )
    # map_results = compute_retrieval_template(
    #     bbdd_images, qsd1_images, my_hsv_block_hist_concat, gt_qsd1_w1, "qsd1_w1"
    # )
    # print(
    #     f"WEEK 2: hsv_block_hist_concat bins={week2_best_bins} grid={week2_best_grid}, qsd1_w1"
    # )
    # print(pd.DataFrame(map_results))

    # WEEK 2 best method execution on qsd2_w2
    # map_results = compute_retrieval_template(
    #     bbdd_images, qsd2_images, my_hsv_block_hist_concat, gt_qsd2_w2, "qsd2_w2"
    # )
    # print(
    #     f"WEEK 2: hsv_block_hist_concat bins={week2_best_bins} grid={week2_best_grid}, qsd2_w2"
    # )
    # print(pd.DataFrame(map_results))

    # WEEK 2 best method execution on qsd1_w3
    map_results = compute_retrieval_template(
        bbdd_images, qsd1_3_images, my_hsv_block_hist_concat, gt_qsd1_w3, "qsd1_w3"
    )
    print(
        f"WEEK 2: hsv_block_hist_concat bins={week2_best_bins} grid={week2_best_grid}, qsd1_w3"
    )
    print(pd.DataFrame(map_results), "\n")

    # WEEK 3 best method execution on qsd1_w3
    my_glcm_block_descriptor = glcm_block_descriptor_func(
        grid=week3_best_glcm_grid,
        distances=week3_best_dist,
        levels=week3_best_lvl,
    )
    map_results = compute_retrieval_template(
        bbdd_images, qsd1_3_images, my_glcm_block_descriptor, gt_qsd1_w3, "qsd1_w3"
    )
    print(
        f"WEEK 3: GLCM grid={week3_best_glcm_grid} distances={week3_best_dist} levels={week3_best_lvl}, qsd1_w3"
    )
    print(pd.DataFrame(map_results), "\n")
    my_dct_block_descriptor = dct_block_descriptor_func(
        n_coefs=week3_best_ncoefs,
        grid=week3_best_grid,
        relative_coefs=False,
    )
    map_results = compute_retrieval_template(
        bbdd_images, qsd1_3_images, my_dct_block_descriptor, gt_qsd1_w3, "qsd1_w3"
    )
    print(f"WEEK 3: DCT grid={week3_best_grid} n_coefs={week3_best_ncoefs}, qsd1_w3")
    print(pd.DataFrame(map_results), "\n")

    # WEEK 3 best method execution on qsd2_w3 crops and perfect crops
    map_results = compute_retrieval_template(
        bbdd_images, qsd2_3_images, my_glcm_block_descriptor, gt_qsd2_w3, "qsd2_w3"
    )
    print(
        f"WEEK 3: GLCM grid={week3_best_glcm_grid} distances={week3_best_dist} levels={week3_best_lvl}, qsd2_w3"
    )
    print(pd.DataFrame(map_results), "\n")
    map_results = compute_retrieval_template(
        bbdd_images,
        qsd2_3_perfect_images,
        my_glcm_block_descriptor,
        gt_qsd2_w3,
        "qsd2_w3_perfect",
    )
    print(
        f"WEEK 3: GLCM grid={week3_best_glcm_grid} distances={week3_best_dist} levels={week3_best_lvl}, qsd2_w3_perfect"
    )
    print(pd.DataFrame(map_results), "\n")
    map_results = compute_retrieval_template(
        bbdd_images, qsd2_3_images, my_dct_block_descriptor, gt_qsd2_w3, "qsd2_w3"
    )
    print(f"WEEK 3: DCT grid={week3_best_grid} n_coefs={week3_best_ncoefs}, qsd2_w3")
    print(pd.DataFrame(map_results), "\n")
    map_results = compute_retrieval_template(
        bbdd_images,
        qsd2_3_perfect_images,
        my_dct_block_descriptor,
        gt_qsd2_w3,
        "qsd2_w3_perfect",
    )
    print(
        f"WEEK 3: DCT grid={week3_best_grid} n_coefs={week3_best_ncoefs}, qsd2_w3_perfect"
    )
    print(pd.DataFrame(map_results), "\n")

    # WEEK 3 generate results for qst1_w3
    compute_test_retrieval(
        bbdd_images,
        qst1_images,
        my_dct_block_descriptor,
        compute_hellinger_distance,
        "qst1_w3",
    )
    # WEEK 3 generate results for qst2_w3
    compute_test_retrieval(
        bbdd_images,
        qst2_images,
        my_dct_block_descriptor,
        compute_hellinger_distance,
        "qst2_w3",
    )


# main function adapted for week 1&2 comparisons:
# 1) load all images in bgr
# 2) preprocess all images
# 3) compute week 1 method as reference point
# 4) compute week 2 methods with different parameters (nested loops, takes a long time to compute)
def test_weekn_weekm(weekn: int = 2, weekm: int = 3):
    print(f"Testing week {weekn} vs week {weekm} methods...")
    run_block_histogram_concat = True
    run_lbp_descriptor = True
    run_dct_descriptor = True
    run_glcm_descriptor = True

    force_retrieval = False  # If True, forces recomputation of descriptors and retrieval even if result pkl files exist
    save_results = True  # If True, saves results of retrieval in method_bins_grids.pkl
    test_mode = False  # If True, only process a few images for quick testing & visualization purposes
    testing_bins = [[4, 4, 2]]
    testing_grids = [(9, 9)]
    week3_bins = [4, 16, 32, 64, 128, 257]
    week3_grids = [(3, 3), (5, 5), (7, 7), (11, 11)]
    relative_coefs = False  # If True, n_coefs is interpreted as percentage of total coefficients in block
    if relative_coefs:
        n_coefs_list = [25, 50, 75, 100]  # Number of DCT coefficients to use
    else:
        n_coefs_list = [10, 20, 30, 40, 50]  # Number of DCT coefficients to use
    lbp_points = [8, 16, 24]  # Number of LBP points to use
    lbp_radius = [1, 2, 3]  # Radius for LBP
    glcm_grids = [
        (1, 1),
        (2, 2),
        (4, 4),
        (6, 6),
        (8, 8),
        (10, 10),
        (12, 12),
        (14, 14),
        (16, 16),
    ]
    glcm_distances = [[1, 2], [1, 2, 3], [1, 3, 5]]
    glcm_levels = [256, 128, 64, 32, 16, 8]

    # Create results directory if it doesn't exist
    if save_results:
        Path("./df_results/no_preprocess_no_resize").mkdir(parents=True, exist_ok=True)

    # 1) Load all images paths
    if test_mode:
        qsd1_3_pathlist = list(
            Path(Path(__file__).parent / "datasets" / "qsd1_w3").glob("*.jpg")
        )[:5]
        qsd1_3_original_pathlist = list(
            Path(Path(__file__).parent / "datasets" / "qsd1_w3" / "non_augmented").glob(
                "*.jpg"
            )
        )[:5]
    else:
        qsd1_3_pathlist = list(
            Path(Path(__file__).parent / "datasets" / "qsd1_w3").glob("*.jpg")
        )
        qsd1_3_original_pathlist = list(
            Path(Path(__file__).parent / "datasets" / "qsd1_w3" / "non_augmented").glob(
                "*.jpg"
            )
        )

    bbdd_pathlist = list(
        Path(Path(__file__).parent / "datasets" / "BBDD").glob("*.jpg")
    )

    # Load ground truth correspondences
    gt_qsd1_w3 = pickle.load(open("./datasets/qsd1_w3/gt_corresps.pkl", "rb"))

    # 1.1) Load all images into a dictionary, key is the filename without extension and value is the image in bgr
    bbdd_images = {
        img_path.stem: cv2.imread(str(img_path)) for img_path in bbdd_pathlist
    }
    qsd1_3_images = {
        img_path.stem: cv2.imread(str(img_path)) for img_path in qsd1_3_pathlist
    }
    qsd1_3_original_images = {
        img_path.stem: cv2.imread(str(img_path))
        for img_path in qsd1_3_original_pathlist
    }

    def to_rgb01(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return rgb

    def to_Y01(bgr):
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        Y = ycrcb[:, :, 0].astype(np.float32) / 255.0
        return Y

    # WEEK 3 TASK 1: NOISE FILTER EVALUATION
    filtered_images = copy.deepcopy(qsd1_3_images)
    preprocess_images(filtered_images)
    vPSNR = []
    avgPSNR = 0
    vSSIM = []
    avgSSIM = 0
    for name, filt_bgr in filtered_images.items():
        ref_bgr = qsd1_3_original_images[name]

        # Mateix resize per a tots dos
        ref_bgr = cv2.resize(ref_bgr, (256, 256), interpolation=cv2.INTER_AREA)
        filt_bgr = cv2.resize(filt_bgr, (256, 256), interpolation=cv2.INTER_AREA)

        # (Opcional) balanç de blancs idèntic a tots dos si vols fer-lo part del preproc
        def gray_world(bgr):
            f = bgr.astype(np.float32) + 1e-6
            mB, mG, mR = [f[:, :, c].mean() for c in range(3)]
            g = (mB + mG + mR) / 3.0
            f[:, :, 0] *= g / mB
            f[:, :, 1] *= g / mG
            f[:, :, 2] *= g / mR
            return np.clip(f, 0, 255).astype(np.uint8)

        # Exemple: aplica’l a tots dos (si el teu pipeline el fa servir)
        ref_bgr = gray_world(ref_bgr)
        filt_bgr = gray_world(filt_bgr)

        ref = to_Y01(ref_bgr)
        out = to_Y01(filt_bgr)

        # SSIM multicanal (mitjana sobre canals)
        s = ssim_metric(ref, out, data_range=1.0)
        if s < 0.8:
            print(f"❌ {name}: SSIM = {s:.3f}")
        vSSIM.append(s)
        avgSSIM += s

        # PSNR amb imatges [0,1]
        mse = np.mean((ref - out) ** 2)
        p = float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)
        vPSNR.append(p)
        avgPSNR += p

    avgSSIM /= len(filtered_images)
    avgPSNR /= len(filtered_images)
    # print(f"Avergae PSNR: {avgPSNR}")
    # print(f"Average SSIM: {avgSSIM}")

    # --- PSNR Plot ---
    plt.figure(figsize=(8, 4))
    plt.plot(vPSNR, marker="o", linestyle="-")
    plt.title("PSNR Values Across Images")
    plt.xlabel("Image Index")
    plt.ylabel("PSNR (dB)")
    plt.show()

    # --- SSIM Plot ---
    plt.figure(figsize=(8, 4))
    plt.plot(vSSIM, marker="o", linestyle="-")
    plt.title("SSIM Values Across Images")
    plt.xlabel("Image Index")
    plt.ylabel("SSIM value")
    plt.show()
    # 2) Preprocess all images with the same preprocessing method (resize 256x256, color balance, contrast&brightness adjustment, smoothing)
    lists_to_preprocess = [bbdd_images, qsd1_3_images, qsd1_3_original_images]
    for i in range(len(lists_to_preprocess)):
        lists_to_preprocess[i] = preprocess_images(lists_to_preprocess[i])

    # WEEK 2 best method: hsv_block_hist_concat
    if run_block_histogram_concat:
        bins = testing_bins[0]
        grid = testing_grids[0]
        # START method hsv_block_hist_concat
        result_pkl_filename = (
            Path("./df_results")
            / "qsd1_w3"
            / f"hsv_block_hist_concat_bins_{bins[0]}-{bins[1]}-{bins[2]}_grid_{grid[0]}-{grid[1]}.pkl"
        )
        if result_pkl_filename.exists() and not force_retrieval:
            map_results = pickle.load(open(result_pkl_filename, "rb"))
            print(f"WEEK 2: hsv_block_hist_concat bins={bins} grid={grid}")
            print(pd.DataFrame(map_results))
        else:
            my_hsv_block_hist_concat = hsv_block_hist_concat_func(bins=bins, grid=grid)
            map_results = compute_retrieval_template(
                bbdd_images,
                qsd1_3_original_images,
                my_hsv_block_hist_concat,
                gt_qsd1_w3,
                "qsd1_w3",
                visualize_output=True,
            )
            print(f"WEEK 2: hsv_block_hist_concat bins={bins} grid={grid}")
            print(pd.DataFrame(map_results))
            print()
            if save_results:
                pickle.dump(map_results, open(result_pkl_filename, "wb"))
            # END method hsv_block_hist_concat

    # WEEK 3 methods: LBP & DCT & GLCM
    if run_lbp_descriptor:
        for points in lbp_points:
            for radius in lbp_radius:
                for bins in week3_bins:
                    # START method LBP
                    result_pkl_filename = (
                        Path("./df_results")
                        / "qsd1_w3"
                        / f"lbp_points_{points}_radius_{radius}.pkl"
                    )
                    if result_pkl_filename.exists() and not force_retrieval:
                        map_results = pickle.load(open(result_pkl_filename, "rb"))
                        print(f"WEEK 3: LBP points={points} radius={radius}")
                        print(pd.DataFrame(map_results))
                        continue
                    my_lbp_descriptor = lbp_descriptor_histogram_func(
                        lbp_p=points, lbp_r=radius, bins=bins
                    )
                    map_results = compute_retrieval_template(
                        bbdd_images,
                        qsd1_3_original_images,
                        my_lbp_descriptor,
                        gt_qsd1_w3,
                        "qsd1_w3",
                        visualize_output=True,
                    )
                    print(f"WEEK 3: LBP points={points} radius={radius}")
                    print(pd.DataFrame(map_results))
                    print()
                    if save_results:
                        pickle.dump(map_results, open(result_pkl_filename, "wb"))
                    # END method LBP
    if run_dct_descriptor:
        for n_coefs in n_coefs_list:
            for grid in week3_grids:
                # START method DCT
                result_pkl_filename = (
                    Path("./df_results")
                    / "qsd1_w3"
                    / f"dct_ncoefs_{n_coefs}_grid_{grid[0]}-{grid[1]}.pkl"
                )
                if result_pkl_filename.exists() and not force_retrieval:
                    map_results = pickle.load(open(result_pkl_filename, "rb"))
                    print(
                        f"WEEK 3: DCT n_coefs={n_coefs} grid={grid} relative_coefs={relative_coefs}"
                    )
                    print(pd.DataFrame(map_results))
                    continue
                my_dct_block_descriptor = dct_block_descriptor_func(
                    n_coefs=n_coefs, grid=grid, relative_coefs=relative_coefs
                )
                map_results = compute_retrieval_template(
                    bbdd_images,
                    qsd1_3_original_images,
                    my_dct_block_descriptor,
                    gt_qsd1_w3,
                    "qsd1_3",
                    visualize_output=True,
                )
                print(
                    f"WEEK 3: DCT n_coefs={n_coefs} grid={grid} relative_coefs={relative_coefs}"
                )
                print(pd.DataFrame(map_results))
                print()
                if save_results:
                    pickle.dump(map_results, open(result_pkl_filename, "wb"))
                # END method DCT
    if run_glcm_descriptor:
        for grid in glcm_grids:
            for distances in glcm_distances:
                for levels in glcm_levels:
                    # START method GLCM
                    result_pkl_filename = (
                        Path("./df_results")
                        / "qsd1_w3"
                        / f"glcm_grid_{grid[0]}-{grid[1]}_distances_{'-'.join(map(str, distances))}_levels_{levels}.pkl"
                    )
                    if result_pkl_filename.exists() and not force_retrieval:
                        map_results = pickle.load(open(result_pkl_filename, "rb"))
                        print(
                            f"WEEK 3: GLCM grid={grid} distances={distances} levels={levels}"
                        )
                        print(pd.DataFrame(map_results))
                        continue
                    my_glcm_block_descriptor = glcm_block_descriptor_func(
                        grid=grid, distances=distances, levels=levels
                    )
                    map_results = compute_retrieval_template(
                        bbdd_images,
                        qsd1_3_original_images,
                        my_glcm_block_descriptor,
                        gt_qsd1_w3,
                        "qsd1_3",
                        visualize_output=True,
                    )
                    print(
                        f"WEEK 3: GLCM grid={grid} distances={distances} levels={levels}"
                    )
                    print(pd.DataFrame(map_results))
                    print()
                    if save_results:
                        pickle.dump(map_results, open(result_pkl_filename, "wb"))
                    # END method GLCM


if __name__ == "__main__":
    main()
