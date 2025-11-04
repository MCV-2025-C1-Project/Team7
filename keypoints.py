import cv2
import matplotlib.pyplot as plt
import numpy as np


def harris_corner_detection(
    img: np.ndarray, blockSize: int = 17, ksize: int = 21, N: int = 2000, visualize: bool = False
) -> list[cv2.KeyPoint]:
    operatedImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    operatedImage = np.float32(operatedImage)

    dest = cv2.cornerHarris(operatedImage, blockSize, ksize, 0.01)  # Try different parameter values
    dest = cv2.dilate(dest, None)

    dst_norm = cv2.normalize(dest, None, 0, 255, cv2.NORM_MINMAX)
    responses = dst_norm.flatten()
    coords = np.column_stack(np.unravel_index(np.argsort(responses)[::-1], dest.shape))
    # Limit to top N points for faster matching. I guess play with this number
    corners = coords[:N]

    keypoints = [(x, y) for y, x in corners]

    if visualize:
        img[dest > 0.01 * dest.max()] = [0, 0, 255]
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
        plt.axis("off")
        plt.show()

    keypoints = to_keypoints(keypoints, size=3)

    return keypoints


def harris_laplacian_detection(img: np.ndarray, visualize: bool = False) -> list[cv2.KeyPoint]:
    operatedImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scales = [1.2, 2, 4, 8, 12, 16, 20]
    k = 0.04
    keypoints = []

    for sigma in scales:
        # Smooth image at this scale
        blur = cv2.GaussianBlur(operatedImage, (0, 0), sigma)
        # Compute Harris response
        Ix = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
        Ixx = Ix**2
        Iyy = Iy**2
        Ixy = Ix * Iy
        Sxx = cv2.GaussianBlur(Ixx, (0, 0), sigma)
        Syy = cv2.GaussianBlur(Iyy, (0, 0), sigma)
        Sxy = cv2.GaussianBlur(Ixy, (0, 0), sigma)
        detM = Sxx * Syy - Sxy**2
        traceM = Sxx + Syy
        R = detM - k * traceM**2

        # Threshold & record keypoints
        corners = np.argwhere(R > 0.01 * R.max())
        for y, x in corners:
            keypoints.append((x, y, sigma, R[y, x]))

    # Keep local maxima in scale-space (approximate)
    keypoints = sorted(keypoints, key=lambda p: p[3], reverse=True)

    if visualize:
        for x, y, sigma, _ in keypoints[:500]:
            cv2.circle(img, (x, y), int(sigma), (0, 0, 255), 1)

        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

    keypoints = to_keypoints(keypoints, size=3)

    return keypoints


def dog_detection(img: np.ndarray) -> list[cv2.KeyPoint]:
    # SIFT is DoG-based
    sift = cv2.SIFT_create()
    keypoints = sift.detect(img, None)
    img_out = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
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


def compute_local_descriptors(
    img: np.ndarray, keypoints: list[cv2.KeyPoint], method="SIFT"
) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    method = method.upper()
    if method == "SIFT":
        extractor = cv2.SIFT_create()
    elif method == "ORB":
        extractor = cv2.ORB_create()
    elif method == "AKAZE":
        extractor = cv2.AKAZE_create()
    else:
        raise ValueError("Method must be one of: 'SIFT', 'ORB', 'AKAZE'")

    keypoints, descriptors = extractor.compute(img, keypoints)
    return keypoints, descriptors


def keypoint_descriptor_template(
    img: np.ndarray, keypoint_func, descriptor_method: str
) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    keypoints = keypoint_func(img)
    return compute_local_descriptors(img, keypoints, method=descriptor_method)


def calculate_matches(desc1: np.ndarray, desc2: np.ndarray, method="BF") -> tuple[int, list[cv2.DMatch]]:
    # Binary descriptors are uint8
    norm = cv2.NORM_L2
    if desc1.dtype == np.uint8:
        norm = cv2.NORM_HAMMING

    method = method.upper()
    if method == "BF":
        matcher = cv2.BFMatcher(norm, crossCheck=True)
    elif method == "FLANN":
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        raise ValueError("Method must be one of: 'BF', 'FLANN'")

    # matches = matcher.knnMatch(desc1, desc2, k=2)
    matches = []
    if desc1.shape[0] < desc2.shape[0]:
        matches = matcher.match(desc2, desc1)
    else:
        matches = matcher.match(desc1, desc2)
    return len(matches), matches

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return len(good_matches)


def keypoint_retrieval(
    bbdd_images: dict[str, list[np.ndarray]],
    query_images: dict[str, list[np.ndarray]],
    keypoint_func,
    descriptor_method: str = "SIFT",
    matcher_method: str = "BF",
    top_k: int = 10,
) -> dict[int, list[list[tuple[int, int]]]]:
    """
    Perform image retrieval using keypoint matching.

    Args:
        bbdd_images: Database images dictionary {name: [image]}
        query_images: Query images dictionary {name: [image1, image2, ...]}
        keypoint_func: Function to detect keypoints (e.g., harris_corner_detection)
        descriptor_method: Local descriptor method ('SIFT', 'ORB', 'AKAZE')
        matcher_method: Matching method ('BF', 'FLANN')
        top_k: Number of top matches to return

    Returns:
        Dictionary mapping query index to list of lists of (n_matches, bbdd_index) tuples
    """
    # Create index mappings
    bbdd_trans = {int(k.split("_")[-1]): k for k in bbdd_images.keys()}
    bbdd_trans_inv = {v: k for k, v in bbdd_trans.items()}
    query_trans = {int(k): k for k in query_images.keys()}
    query_trans_inv = {v: k for k, v in query_trans.items()}

    # Compute keypoints and descriptors for all BBDD images
    print("Computing BBDD keypoints and descriptors...")
    bbdd_descriptors = {}
    bbdd_keypoints = {}
    for name, img_list in bbdd_images.items():
        img = img_list[0]
        kps = keypoint_func(img.copy(), visualize=False)
        _, desc = compute_local_descriptors(img, kps, method=descriptor_method)
        bbdd_descriptors[name] = desc
        bbdd_keypoints[name] = kps

    # Compute keypoints and descriptors for all query images
    print("Computing query keypoints and descriptors...")
    query_descriptors = {}
    query_keypoints = {}
    for name, img_list in query_images.items():
        query_descriptors[name] = []
        query_keypoints[name] = []
        for img in img_list:
            kps = keypoint_func(img.copy(), visualize=False)
            _, desc = compute_local_descriptors(img, kps, method=descriptor_method)
            query_descriptors[name].append(desc)
            query_keypoints[name].append(kps)

    # Perform matching and retrieval
    print("Performing matching and retrieval...")
    results = {}
    for query_name, query_desc_list in query_descriptors.items():
        query_idx = query_trans_inv[query_name]
        results[query_idx] = []

        for query_desc in query_desc_list:
            matches_list = []

            for bbdd_name, bbdd_desc in bbdd_descriptors.items():
                bbdd_idx = bbdd_trans_inv[bbdd_name]
                n_matches, _ = calculate_matches(query_desc, bbdd_desc, method=matcher_method)
                matches_list.append((n_matches, bbdd_idx))

            # Sort by number of matches (descending) and take top_k
            matches_list.sort(key=lambda x: x[0], reverse=True)
            results[query_idx].append(matches_list[:top_k])

    return results
