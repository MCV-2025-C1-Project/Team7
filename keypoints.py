import cv2
import matplotlib.pyplot as plt
import numpy as np


def harris_corner_detection(
    img: np.ndarray, blockSize: int = 17, ksize: int = 21, visualize: bool = False
) -> list[cv2.KeyPoint]:
    operatedImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    operatedImage = np.float32(operatedImage)

    dest = cv2.cornerHarris(operatedImage, blockSize, ksize, 0.01)  # Try different parameter values
    dest = cv2.dilate(dest, None)

    pts = np.argwhere(dest > 0.01 * dest.max())
    keypoints = [(x, y) for y, x in pts]

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


def calculateMatches(desc1: np.ndarray, desc2: np.ndarray, method="BF") -> list[cv2.DMatch]:
    method = method.upper()
    if method == "BF":
        matcher = cv2.BFMatcher()
    elif method == "FLANN":
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        raise ValueError("Method must be one of: 'BF', 'FLANN'")

    matches = matcher.knnMatch(desc1, desc2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return good_matches