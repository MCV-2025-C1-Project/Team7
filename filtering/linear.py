import numpy as np

def mean_filter(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Args:
        img: A 2D numpy array representing a grayscale image.
        kernel_size: size of the square kernel.
    Returns:
        The filtered image
    """
    h, w = img.shape
    padding = kernel_size // 2
    padded_img = np.pad(img, padding, mode="edge")
    out = np.zeros(img.shape, dtype=int)

    for y in range(h):
        for x in range(w):
            sliding_window = padded_img[y:y+kernel_size, x:x+kernel_size]
            out[y, x] = np.mean(sliding_window)
            
    return out