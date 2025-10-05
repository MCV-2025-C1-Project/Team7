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


#ADVO INI : Laplacian Filter
def laplacian_filter(image: np.ndarray) -> np.ndarray:
    """
    Aplica un filtro Laplaciano manual para detectar bordes en una imagen en escala de grises.
    Implementación sin usar convolve2d.
    Args:
        image: Imagen 2D (grayscale) como un array numpy.
    Returns:
        Imagen filtrada (bordes resaltados) como un array numpy.
    """

    kernel = np.array([
        [0,  1,  0],
        [1, -4,  1],
        [0,  1,  0]
    ])

    image = image.astype(np.float32)

    padded = np.pad(image, pad_width=1, mode='reflect')

    output = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+3, j:j+3]
            output[i, j] = np.sum(region * kernel)

    output = np.clip(output, 0, 255).astype(np.uint8)

    return output

#ADVO FIN : Laplacian Filter