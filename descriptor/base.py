from collections.abc import Callable

import numpy as np
import tqdm
from pathlib import Path
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

def cumsum(a):
    """
    Computes a cumulated histogram.
    Args:
        image: A 2D numpy array representing a grayscale image.
    Returns:
        A 1D numpy array of length 256 representing the cumulated histogram.
    """
    a = iter(a)
    b = [next(a)]
    for i in a:
        b.append(b[-1] + i)
    return np.array(b)

def equalization(img: np.ndarray) -> np.ndarray:
    """
    Computes the equalized image.
    Args:
        img: A 2D numpy array representing a grayscale image.
    Returns:
        A 2D numpy array representing the equalized image.
    """
    cs = cumsum(grayscale_histogram(img))
    nj = (cs - cs.min()) * 255
    N = cs.max() - cs.min()
    
    # re-normalize the cumsum
    cs = nj / N
    
    # cast it back to uint8 since we can't use floating point values in images
    cs = cs.astype('uint8')
    
    # get the value from cumulative sum for every index in flat, and set that as img_new
    img_new = cs[img.flatten()]
    
    # put array back into original shape since we flattened it
    img_new = np.reshape(img_new, img.shape)
    
    return img_new


def compute_descriptors(
    suffix: str,
    method: Callable[[np.ndarray], np.ndarray],
    images: dict[str, np.ndarray],
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
    pkl_path = Path(__file__).parent / f"{suffix}-{method.__name__}-{('grayscale' if use_grayscale else 'rgb')}.pkl"
    if pkl_path.exists() and not overwrite_pkl:
        with open(pkl_path, "rb") as f:
            descriptors = pickle.load(f)
    else:
        descriptors = {}
        for imgname in tqdm.tqdm(images.keys(), desc="Computing descriptors"):
            if use_grayscale:
                image = np.array(Image.open(imgname).convert("L"))
            else:
                image = np.array(Image.open(imgname).convert("RGB"))
            descriptor = method(image)
            descriptor = method(image)
            descriptor_index = int(imgname.split("_")[-1])
            descriptors[descriptor_index] = descriptor
        if save_as_pkl:
            with open(pkl_path, "wb") as f:
                pickle.dump(descriptors, f)
    return descriptors
