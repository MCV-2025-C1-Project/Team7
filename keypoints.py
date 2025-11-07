import pickle
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter
from tqdm import tqdm


def harris_corner_detection(
    img: np.ndarray,
    blockSize: int = 2,
    ksize: int = 3,
    k: float = 0.04,
    threshold: float = 0.01,
    nms_radius: int = 5,
    visualize: bool = False,
) -> list[cv2.KeyPoint]:
    """
    Harris corner detection with adaptive thresholding and non-maximum suppression.

    Args:
        img: Input BGR image
        blockSize: Neighborhood size for corner detection (2, 3, 5, 7)
        ksize: Aperture parameter for Sobel operator (3, 5, 7)
        k: Harris detector free parameter (0.04-0.06)
        threshold: Threshold for corner response (0.001-0.1, relative to max response)
        nms_radius: Radius for non-maximum suppression (3-10 pixels)
        visualize: Whether to visualize detected corners

    Returns:
        List of cv2.KeyPoint objects
    """
    operatedImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    operatedImage = np.float32(operatedImage)

    dest = cv2.cornerHarris(operatedImage, blockSize, ksize, k)  # Try different parameter values
    dest = cv2.dilate(dest, None)

    threshold_value = threshold * dest.max()
    corner_mask = dest > threshold_value

    local_max = maximum_filter(dest, size=2 * nms_radius + 1)
    nms_mask = (dest == local_max) & corner_mask

    coords = np.argwhere(nms_mask)
    responses = dest[nms_mask]

    keypoints = [(x, y, r) for (y, x), r in zip(coords, responses)]
    keypoints.sort(key=lambda p: p[2], reverse=True)

    if visualize:
        savepath = Path("./visualize") / "harris_corner_detection"
        savepath.mkdir(parents=True, exist_ok=True)
        vis_img = img.copy()
        for x, y, r in keypoints[:500]:  # Show top 500 for clarity
            cv2.circle(vis_img, (int(x), int(y)), 3, (0, 0, 255), -1)

        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Harris Corners: {len(keypoints)} detected")
        plt.axis("off")
        plt.savefig(savepath / "harris_corners.png")
        plt.close()

    keypoints = to_keypoints(keypoints, size=3)

    return keypoints


def harris_corner_detection_func(
    blockSize: int = 2,
    ksize: int = 3,
    k: float = 0.04,
    threshold: float = 0.01,
    nms_radius: int = 5,
    visualize: bool = False,
) -> Callable[[np.ndarray], list[cv2.KeyPoint]]:
    def custom_harris_corner_detection(img: np.ndarray) -> list[cv2.KeyPoint]:
        return harris_corner_detection(
            img,
            blockSize=blockSize,
            ksize=ksize,
            k=k,
            threshold=threshold,
            nms_radius=nms_radius,
            visualize=visualize,
        )

    return custom_harris_corner_detection

def harris_laplacian_detection_func(
    blockSize: int = 3,
    ksize: int = 3,
    k: float = 0.06,
    threshold: float = 0.001,
    nms_radius: int = 3,
    max_kps: int = 2000,
    visualize: bool = False,
    **params
) -> Callable[[np.ndarray], list[cv2.KeyPoint]]:
    def custom_harris_laplacian_detection(img: np.ndarray) -> list[cv2.KeyPoint]:
        return harris_laplacian_detection(
            img,
            blockSize=blockSize,
            ksize=ksize,
            k=k,
            threshold=threshold,
            nms_radius=nms_radius,
            visualize=visualize,
            **params
        )

    return custom_harris_laplacian_detection

def harris_laplacian_detection(
    img: np.ndarray,
    blockSize: int = 3,
    ksize: int = 3,
    k: float = 0.06,
    threshold: float = 0.001,
    nms_radius: int = 3,
    max_kps: int = 2000,
    visualize: bool = False,
    **params

) -> list[cv2.KeyPoint]:
    """
    Harris–Laplacian keypoint detector.

    Combines Harris corner detection for spatial localization
    and Laplacian-based scale selection for scale invariance.
    """
    scales = params['scales']
    
    # Ensure grayscale float
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    gray = np.float32(gray) / 255.0

    all_responses = []

    for sigma in scales:
        # Gaussian smoothing
        blur = cv2.GaussianBlur(gray, (0, 0), sigma)

        # Harris corner measure
        Ix = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=ksize)
        Iy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=ksize)
        Ixx, Iyy, Ixy = Ix**2, Iy**2, Ix * Iy
        Sxx = cv2.GaussianBlur(Ixx, (blockSize, blockSize), sigma)
        Syy = cv2.GaussianBlur(Iyy, (blockSize, blockSize), sigma)
        Sxy = cv2.GaussianBlur(Ixy, (blockSize, blockSize), sigma)
        detM = Sxx * Syy - Sxy**2
        traceM = Sxx + Syy
        R = detM - k * traceM**2

        # Laplacian-of-Gaussian (for scale selection)
        LoG = np.abs(sigma**2 * cv2.Laplacian(blur, cv2.CV_32F))

        all_responses.append((R, LoG, sigma))

    # Collect keypoints with spatial + scale-space NMS
    keypoints = []
    for i, (R, LoG, sigma) in enumerate(all_responses):
        # Threshold on Harris response
        R_thresh = R > threshold * R.max()
        coords = np.argwhere(R_thresh)

        # 2D Non-maximum suppression
        for y, x in coords:
            y0, y1 = max(0, y - nms_radius), min(R.shape[0], y + nms_radius + 1)
            x0, x1 = max(0, x - nms_radius), min(R.shape[1], x + nms_radius + 1)
            local_patch = R[y0:y1, x0:x1]

            if R[y, x] < np.max(local_patch):
                continue  # Not spatial maximum

            # Check scale-space maximum (LoG)
            prev_LoG = all_responses[i-1][1][y, x] if i > 0 else -np.inf
            next_LoG = all_responses[i+1][1][y, x] if i < len(all_responses)-1 else -np.inf
            if LoG[y, x] > prev_LoG and LoG[y, x] > next_LoG:
                keypoints.append(cv2.KeyPoint(float(x), float(y), size=float(2*sigma)))
                
    if len(keypoints) > 0:
        # Extract responses for each keypoint
        responses = []
        for kp in keypoints:
            y, x = int(kp.pt[1]), int(kp.pt[0])
            # Pick corresponding R value at nearest scale
            R_nearest = all_responses[min(range(len(all_responses)),
                                          key=lambda i: abs(all_responses[i][2] - kp.size/2))][0]
            kp.response = R_nearest[y, x]
            responses.append(kp.response)
    
        # Sort and keep only the strongest max_kps
        keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)
        keypoints = keypoints[:max_kps]

    # Visualization
    if visualize:
        img_viz = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for kp in keypoints:
            cv2.circle(img_viz, (int(kp.pt[0]), int(kp.pt[1])), int(kp.size/2), (0, 0, 255), 1)
        plt.imshow(cv2.cvtColor(img_viz, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("Harris–Laplacian Keypoints")
        plt.show()

    return keypoints


def dog_detection(img: np.ndarray, visualize: bool = False) -> list[cv2.KeyPoint]:
    if (len(img.shape) > 2):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # SIFT is DoG-based
    sift = cv2.SIFT_create()
    keypoints = sift.detect(img, None)
    img_out = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    if visualize:
        plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_GRAY2BGR))
        plt.axis("off")
        plt.show()

    return keypoints


def to_keypoints(points, size=3):
    kps = []
    for p in points:
        if len(p) == 4:
            x, y, s, r = p
            kp = cv2.KeyPoint(float(x), float(y), float(s))
            kp.response = float(r)
            kps.append(kp)
        elif len(p) == 3:
            x, y, s = p
            kps.append(cv2.KeyPoint(float(x), float(y), float(s)))
        else:
            x, y = p
            kps.append(cv2.KeyPoint(float(x), float(y), float(size)))
    return kps


def opponent_sift(image, keypoints=None, max_points=None):
    if (len(image.shape) == 2):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    img = image.astype(np.float32) / 255.0
    B, G, R = cv2.split(img)

    # Opponent channels
    O1 = (R - G) / np.sqrt(2.0)
    O2 = (R + G - 2.0 * B) / np.sqrt(6.0)
    O3 = (R + G + B) / np.sqrt(3.0)

    # Normalize each channel to [0,255] and convert to uint8
    def normalize_channel(ch):
        ch_norm = cv2.normalize(ch, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        return ch_norm.astype(np.uint8)

    O1 = normalize_channel(O1)
    O2 = normalize_channel(O2)
    O3 = normalize_channel(O3)

    # Create SIFT
    sift = cv2.SIFT_create()

    # If no keypoints given, detect them on grayscale
    if keypoints is None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints = sift.detect(gray, None)

    if max_points and len(keypoints) > max_points:
        keypoints = sorted(keypoints, key=lambda k: k.response, reverse=True)[:max_points]

    # Compute descriptors per channel
    _, desc1 = sift.compute(O1, keypoints)
    _, desc2 = sift.compute(O2, keypoints)
    _, desc3 = sift.compute(O3, keypoints)

    if desc1 is None or desc2 is None or desc3 is None:
        return keypoints, None

    descriptors = np.hstack([desc1, desc2, desc3])
    return keypoints, descriptors


def compute_local_descriptors(
    img: np.ndarray, keypoints: list[cv2.KeyPoint], method="SIFT"
) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    method = method.upper()
    
    descriptors = None
    if method == "SIFT":
        extractor = cv2.SIFT_create()
        _, descriptors = extractor.compute(img, keypoints)
    elif method == "ORB":
        extractor = cv2.ORB_create()
        _, descriptors = extractor.compute(img, keypoints)
    elif method == "COLOR-SIFT":
        _, descriptors = opponent_sift(img, keypoints)
    else:
        raise ValueError("Method must be one of: 'SIFT', 'ORB', 'COLOR-SIFT'")
        
    return descriptors


def keypoint_descriptor_template(
    img: np.ndarray, keypoint_func, descriptor_method: str
) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    keypoints = keypoint_func(img)
    return compute_local_descriptors(img, keypoints, method=descriptor_method)


def calculate_matches(
    desc1: np.ndarray, desc2: np.ndarray, method="BF", ratio_threshold=0.7, use_cross_check=True, use_ratio_test=True
) -> tuple[int, list[cv2.DMatch]]:
    """
    Calculate matches between two descriptor sets with optional ratio test and cross-check.

    Args:
        desc1: First descriptor set
        desc2: Second descriptor set
        method: Matching method ('BF' or 'FLANN')
        ratio_threshold: Lowe's ratio threshold (0.75 recommended)
        use_cross_check: Use cross-check constraint (only for BF, more restrictive)
        use_ratio_test: Apply Lowe's ratio test (better quality, incompatible with crossCheck)

    Returns:
        Tuple of (number of good matches, list of matches)
    """
    # Handle empty descriptors
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return 0, []

    # Binary descriptors are uint8
    norm = cv2.NORM_HAMMING if desc1.dtype == np.uint8 else cv2.NORM_L2

    method = method.upper()
    # CrossCheck and ratio test are mutually exclusive
    if use_cross_check and use_ratio_test:
        use_ratio_test = False  # Prioritize cross-check for BF

    if method == "BF":
        # Use crossCheck for higher quality matches (symmetric matching)
        matcher = cv2.BFMatcher(norm, crossCheck=use_cross_check)

        if use_cross_check:
            # CrossCheck uses simple match() - no ratio test
            matches = matcher.match(desc1, desc2)
            # Sort by distance (lower is better)
            good_matches = sorted(matches, key=lambda x: x.distance)

            # Optional: apply distance threshold for even stricter filtering
            if len(good_matches) > 0:
                # Keep matches within 2x of minimum distance
                min_dist = good_matches[0].distance
                threshold = max(min_dist * 2.5, 30.0)  # At least 30 to avoid being too strict
                good_matches = [m for m in good_matches if m.distance < threshold]

            return len(good_matches), good_matches

        elif use_ratio_test:
            # Use knnMatch for ratio test
            matches = None
            if (desc1.shape[0] > desc2.shape[0]):
                matches = matcher.knnMatch(desc1, desc2, k=2)
            else:
                matches = matcher.knnMatch(desc2, desc1, k=2)
                    
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < ratio_threshold * n.distance:
                        good_matches.append(m)
                elif len(match_pair) == 1:
                    good_matches.append(match_pair[0])

            return len(good_matches), good_matches

        else:
            # No filtering, just match
            matches = matcher.match(desc1, desc2)
            return len(matches), matches

    elif method == "FLANN":
        # FLANN doesn't support crossCheck, always use ratio test
        if desc1.dtype == np.uint8:
            # LSH for binary descriptors
            index_params = dict(
                algorithm=6,  # FLANN_INDEX_LSH
                table_number=6,
                key_size=12,
                multi_probe_level=1,
            )
        else:
            # KDTree for float descriptors
            index_params = dict(algorithm=1, trees=5)

        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)

        if use_ratio_test:
            # Use knnMatch for ratio test
            matches = matcher.knnMatch(desc1, desc2, k=2)
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < ratio_threshold * n.distance:
                        good_matches.append(m)
                elif len(match_pair) == 1:
                    good_matches.append(match_pair[0])

            return len(good_matches), good_matches
        else:
            matches = matcher.match(desc1, desc2)
            return len(matches), matches

    else:
        raise ValueError("Method must be one of: 'BF', 'FLANN'")


def keypoint_retrieval(
    bbdd_images: dict[str, list[np.ndarray]],
    query_images: dict[str, list[np.ndarray]],
    keypoint_func,
    descriptor_method: str = "SIFT",
    matcher_method: str = "BF",
    use_cross_check: bool = True,
    use_ratio_test: bool = False,
    ratio_threshold: float = 0.75,
    top_k: int = 10,
    parallel: bool = False,
    max_workers: int = 8,
    **keypoint_params,
) -> dict[int, list[list[tuple[int, int]]]]:
    """
    Perform image retrieval using keypoint matching with optional parallelization.

    Args:
        bbdd_images: Database images dictionary {name: [image]}
        query_images: Query images dictionary {name: [image1, image2, ...]}
        keypoint_func: Function to detect keypoints (e.g., harris_corner_detection)
        descriptor_method: Local descriptor method ('SIFT', 'ORB', 'AKAZE')
        matcher_method: Matching method ('BF', 'FLANN')
        use_cross_check: Use cross-check constraint (only for BF, higher quality)
        use_ratio_test: Apply Lowe's ratio test (better quality, slower)
        ratio_threshold: Threshold for ratio test
        top_k: Number of top matches to return
        parallel: Use parallel processing for descriptor computation
        max_workers: Number of parallel workers

    Returns:
        Dictionary mapping query index to list of lists of (n_matches, bbdd_index) tuples
    """
    # Create index mappings
    bbdd_trans = {int(k.split("_")[-1]): k for k in bbdd_images.keys()}
    bbdd_trans_inv = {v: k for k, v in bbdd_trans.items()}
    query_trans = {int(k): k for k in query_images.keys()}
    query_trans_inv = {v: k for k, v in query_trans.items()}

    pkl_path = Path("./week4_pkl")
    pkl_path.mkdir(parents=True, exist_ok=True)

    def compute_descriptors_for_image(img):
        kps = keypoint_func(img.copy())
        desc = compute_local_descriptors(img, kps, method=descriptor_method)
        return desc

    # Compute keypoints and descriptors for all BBDD images
    print("Computing BBDD keypoints and descriptors...")
    pkl_path = Path("./week4_pkl") / f"bbdd_{keypoint_func.__name__}_{descriptor_method}.pkl"
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            bbdd_descriptors = pickle.load(f)
        print("Loaded BBDD descriptors from pickle.")
    else:
        bbdd_descriptors = {}

        if parallel:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(compute_descriptors_for_image, img_list[0]): name
                    for name, img_list in bbdd_images.items()
                }
                for future in as_completed(futures):
                    name = futures[future]
                    bbdd_descriptors[name] = future.result()
        else:
            for name, img_list in tqdm(bbdd_images.items(), desc="BBDD images"):
                bbdd_descriptors[name] = compute_descriptors_for_image(img_list[0])
        pickle.dump(bbdd_descriptors, open(pkl_path, "wb"))

    # Compute keypoints and descriptors for all query images
    print("Computing query keypoints and descriptors...")
    pkl_path = Path("./week4_pkl") / f"query_{keypoint_func.__name__}_{descriptor_method}.pkl"
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            query_descriptors = pickle.load(f)
        print("Loaded query descriptors from pickle.")
    else:
        query_descriptors = {}

        if parallel:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for name, img_list in query_images.items():
                    futures = [executor.submit(compute_descriptors_for_image, img) for img in img_list]
                    query_descriptors[name] = [future.result() for future in futures]
        else:
            for name, img_list in tqdm(query_images.items(), desc="Query images"):
                query_descriptors[name] = [compute_descriptors_for_image(img) for img in img_list]
        pickle.dump(query_descriptors, open(pkl_path, "wb")) 
    
    # Perform matching and retrieval
    print(f"Performing matching and retrieval (crossCheck={use_cross_check}, ratioTest={use_ratio_test})...")
    results = {}

    try:
        for query_name, query_desc_list in query_descriptors.items():
            query_idx = query_trans_inv[query_name]
            results[query_idx] = []
        
            for query_desc in query_desc_list:
                matches_list = []
    
                for bbdd_name, bbdd_desc in bbdd_descriptors.items():
                    bbdd_idx = bbdd_trans_inv[bbdd_name]
                    n_matches, _ = calculate_matches(
                        query_desc,
                        bbdd_desc,
                        method=matcher_method,
                        ratio_threshold=ratio_threshold,
                        use_cross_check=use_cross_check,
                        use_ratio_test=use_ratio_test,
                    )
                    matches_list.append((n_matches, bbdd_idx))
    
                # Sort by number of matches (descending) and take top_k
                matches_list.sort(key=lambda x: x[0], reverse=True)
                results[query_idx].append(matches_list[:top_k])
    except Exception as e:
        print(f"Error matching: {e}")

    return results
