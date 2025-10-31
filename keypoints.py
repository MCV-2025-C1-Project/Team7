import numpy as np
import cv2
import matplotlib.pyplot as plt

def harris_corner_detection(img: np.ndarray) -> np.ndarray:
    operatedImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    operatedImage = np.float32(operatedImage)
    
    dest = cv2.cornerHarris(operatedImage, 17, 21, 0.01)  # Try different parameter values
    dest = cv2.dilate(dest, None)
    
    img[dest > 0.01 * dest.max()] = [0, 0, 255]
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    pts = np.argwhere(dest > 0.01 * dest.max())
    keypoints = [(x, y) for y, x in pts] 
    
    plt.imshow(image_rgb)
    plt.axis('off')  
    plt.show()
    
    return keypoints
    
def harris_laplacian_detection(img: np.ndarray) -> np.ndarray: 
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
        Ixy = Ix*Iy
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
    img_out = cv2.cvtColor(operatedImage, cv2.COLOR_GRAY2BGR)
    for (x, y, sigma, _) in keypoints[:500]:
        cv2.circle(img_out, (x, y), int(sigma), (0, 0, 255), 1)
    
    plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    
    return keypoints
    
def dog_detection(img: np.ndarray) -> np.ndarray: 
    # SIFT is DoG-based
    sift = cv2.SIFT_create()
    keypoints = sift.detect(img, None)
    img_out = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
    plt.axis('off')
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

def compute_local_descriptors(img: np.ndarray, keypoints: np.ndarray, method='SIFT'):

    method = method.upper()
    if method == 'SIFT':
        extractor = cv2.SIFT_create()
    elif method == 'ORB':
        extractor = cv2.ORB_create()
    elif method == 'AKAZE':
        extractor = cv2.AKAZE_create()
    else:
        raise ValueError("Method must be one of: 'SIFT', 'ORB', 'AKAZE'")
    
    keypoints, descriptors = extractor.compute(img, keypoints)
    return keypoints, descriptors