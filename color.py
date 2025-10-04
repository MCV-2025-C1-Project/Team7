import numpy as np


def rgb2hsv(rgb: np.ndarray) -> np.ndarray:
    """convert RGB to HSV color space

    :param rgb: np.ndarray
    :return: np.ndarray
    """
    rgb = rgb.astype("float") / 255.0

    maxv = np.max(rgb, axis=2)
    minv = np.min(rgb, axis=2)
    delta = maxv - minv

    hsv = np.zeros_like(rgb)

    mask = delta != 0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    idx = (maxv == r) & mask
    hsv[..., 0][idx] = (60 * ((g - b) / delta % 6))[idx]

    idx = (maxv == g) & mask
    hsv[..., 0][idx] = (60 * ((b - r) / delta + 2))[idx]

    idx = (maxv == b) & mask
    hsv[..., 0][idx] = (60 * ((r - g) / delta + 4))[idx]

    hsv[..., 1][maxv != 0] = (delta / maxv)[maxv != 0]

    hsv[..., 2] = maxv

    return hsv
