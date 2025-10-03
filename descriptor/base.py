from collections.abc import Callable

import numpy as np
import tqdm
import pathlib
from PIL import Image
import pickle


def grayscale_histogram(image: np.ndarray) -> np.ndarray:
    """
    Compute the grayscale histogram of an image.
    Args:
        image: A 2D numpy array representing a grayscale image.
    Returns:
        A 1D numpy array of length 256 representing the histogram.
    """

    histogram = np.zeros(256, dtype=int)
    for pixel in image.flatten():
        histogram[pixel] += 1
    return histogram


def concat_rgb_histogram(image: np.ndarray) -> np.ndarray:
    """
    Compute the concatenated RGB histogram of an image.
    Args:
        image: A 3D numpy array representing an RGB image.
    Returns:
        A 1D numpy array of length 768 representing the concatenated histogram.
    """

    r_hist = np.zeros(256, dtype=int)
    g_hist = np.zeros(256, dtype=int)
    b_hist = np.zeros(256, dtype=int)

    for r, g, b in image.reshape(-1, 3):
        r_hist[r] += 1
        g_hist[g] += 1
        b_hist[b] += 1

    return np.concatenate([r_hist, g_hist, b_hist])


def compute_descriptors(
    method: Callable[[np.ndarray], np.ndarray],
    pathlist: list[pathlib.Path],
    use_grayscale: bool = True,
    save_as_pkl: bool = False,
    overwrite_pkl: bool = False,
) -> dict[int, np.ndarray]:
    """
    Compute descriptors for a list of image paths using the provided method. Loads from .pkl if available.
    If save_as_pkl is True, save the computed descriptors as .pkl files alongside the images.
    Args:
        method: Callable function that accepts an image array and computes the descriptor (returned as a numpy array).
        pathlist: List of pathlib.Path objects pointing to the images.
        use_grayscale: Boolean flag to convert images to grayscale before computing descriptors.
        save_as_pkl: Boolean flag to save computed descriptors as .pkl files.
        overwrite_pkl: Boolean flag to overwrite existing .pkl files if they already exist.
    Returns:
        A dictionary mapping image indices (extracted from filenames) to their computed descriptors.
    """

    descriptors = {}
    for img_path in tqdm.tqdm(pathlist):
        pkl_path = img_path.with_suffix(".pkl")
        if pkl_path.exists() and not overwrite_pkl:
            with open(pkl_path, "rb") as f:
                descriptor = pickle.load(f)
            descriptors[int(img_path.stem.split("_")[-1])] = descriptor
        else:
            if use_grayscale:
                image = np.array(Image.open(img_path).convert("L"))
            else:
                image = np.array(Image.open(img_path).convert("RGB"))
            descriptor = method(image)
            descriptor_index = int(img_path.stem.split("_")[-1])
            descriptors[descriptor_index] = descriptor
            if save_as_pkl:
                with open(pkl_path, "wb") as f:
                    pickle.dump(descriptor, f)
    return descriptors
