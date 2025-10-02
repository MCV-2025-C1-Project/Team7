import numpy as np

def dilate(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Args:
        img: A 2D numpy array representing a grayscale image.
        kernel_size: size of the square kernel.
    Returns:
        The dilated image
    """
    h, w = img.shape
    padding = kernel_size // 2
    padded_img = np.pad(img, padding, mode="edge")
    out = np.zeros(img.shape, dtype=int)

    for y in range(h):
        for x in range(w):
            sliding_window = padded_img[y:y+kernel_size, x:x+kernel_size]
            out[y, x] = np.max(sliding_window)
            
    return out


def erode(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Args:
        img: A 2D numpy array representing a grayscale image.
        kernel_size: size of the square kernel.
    Returns:
        The dilated image
    """
    h, w = img.shape
    padding = kernel_size // 2
    padded_img = np.pad(img, padding, mode="edge")
    out = np.zeros_like(img)

    for y in range(h):
        for x in range(w):
            sliding_window = padded_img[y:y+kernel_size, x:x+kernel_size]
            out[y, x] = np.min(sliding_window)
            
    return out