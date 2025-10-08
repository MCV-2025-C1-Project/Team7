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
