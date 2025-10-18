import numpy as np
from math import log10, sqrt
from skimage.metrics import structural_similarity as ssim
import cv2

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


def binary_mask_evaluation(mask: np.ndarray, gt: np.ndarray):
    """
    Computes the precision, recall and F1-measure of a binary mask
    Args:
        mask: A np array representing the binary mask to evaluate.
        gt: A np array representing the ground truth binary mask.
    Returns:
        A dictionary with the precision, recall and F1-measure values.
    """
    if mask.shape != gt.shape:
        print("Can't compare masks of different sizes")
        return

    mask = mask > 0
    gt = gt > 0

    TP = np.sum(np.logical_and(gt, mask))
    FN = np.sum(np.logical_and(gt, np.logical_not(mask)))
    FP = np.sum(np.logical_and(np.logical_not(gt), mask))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    F1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"precision": precision, "recall": recall, "F1": F1}


def PSNR(original, noisy):
    """
    Computes the PSNR
    Args:
        original: The original image.
        noisy: The image with noise we want to compare.
    Returns:
        The PSNR value.
    """
    mse = np.mean((original - noisy) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))

    return psnr

def SSIM(original, noisy) :
    """
    Computes the structural similarity index
    Args:
        original: The original image.
        noisy: The image with noise we want to compare.
    Returns:
        The SSIM value.
    """
    if original.shape != noisy.shape:
        print("Can't compare masks of different sizes")
        return

    ssim_score, dif = ssim(original, noisy, full=True, channel_axis=2)

    return ssim_score

















