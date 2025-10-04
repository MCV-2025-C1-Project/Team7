from typing import Callable
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
            sliding_window = padded_img[y : y + kernel_size, x : x + kernel_size]
            out[y, x] = np.mean(sliding_window)

    return out


def apply_filters(
    images: dict[str, np.ndarray], filters: list[Callable[[np.ndarray], np.ndarray]]
) -> dict[str, np.ndarray]:
    """
    Applies a series of filters to an image.
    Args:
        img: A 2D numpy array representing a grayscale image.
        filters: A list of filter functions to apply to the image.
    Returns:
        The filtered image.
    """
    for filter_func in filters:
        images = {img_name: filter_func(img) for img_name, img in images.items()}
    return images
