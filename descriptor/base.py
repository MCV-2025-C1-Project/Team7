from collections.abc import Callable

import numpy as np
import tqdm
import pathlib
from PIL import Image, ImageOps
import pickle
import matplotlib.pyplot as plt

from filtering.linear import mean_filter
from morphology.base import openning, closing
from color.base import rgb2hsv

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
    method: Callable[[np.ndarray], np.ndarray],
    pathlist: list[pathlib.Path],
    save_as_pkl: bool = False,
) -> dict[int, np.ndarray]:
    """
    Compute descriptors for a list of image paths using the provided method. Loads from .pkl if available.
    If save_as_pkl is True, save the computed descriptors as .pkl files alongside the images.
    Args:
        method: Callable function that accepts an image array and computes the descriptor (returned as a numpy array).
        pathlist: List of pathlib.Path objects pointing to the images.
    save_as_pkl: Boolean flag to save computed descriptors as .pkl files.
    Returns:
        A dictionary mapping image indices (extracted from filenames) to their computed descriptors.
    """

    descriptors = {}
    for img_path in tqdm.tqdm(pathlist):
        pkl_path = img_path.with_suffix(".pkl")
        # if pkl_path.exists():
        #     with open(pkl_path, "rb") as f:
        #         descriptor = pickle.load(f)
        #     descriptors[int(img_path.stem.split("_")[-1])] = descriptor
        # else:
            
        image = Image.open(img_path)
        
        gray_image = np.array(ImageOps.grayscale(image))
        plt.imshow(gray_image)
        plt.title("gray Image")
        plt.show()
        
        grayscale_smoothed = mean_filter(gray_image)
        plt.imshow(grayscale_smoothed)
        plt.title("gray smoothed Image")
        plt.show()

        rgb_image = np.array(image)
        plt.imshow(rgb_image)
        plt.title("RGB Image")
        plt.show()

        red_channel = rgb_image[:, :, 0]
        green_channel = rgb_image[:, :, 1]
        blue_channel = rgb_image[:, :, 2]
        
        red_channel_smoothed = mean_filter(red_channel)
        green_channel_smoothed = mean_filter(green_channel)
        blue_channel_smoothed = mean_filter(blue_channel)
        rgb_single_smoothed = np.stack((red_channel_smoothed, green_channel_smoothed, blue_channel_smoothed), axis=-1)
        
        red_channel_smoothed = mean_filter(red_channel_smoothed)
        red_channel_smoothed = mean_filter(red_channel_smoothed)
        red_channel_smoothed = mean_filter(red_channel_smoothed)
        
        green_channel_smoothed = mean_filter(green_channel_smoothed)
        green_channel_smoothed = mean_filter(green_channel_smoothed)
        green_channel_smoothed = mean_filter(green_channel_smoothed)
        
        blue_channel_smoothed = mean_filter(blue_channel_smoothed)
        blue_channel_smoothed = mean_filter(blue_channel_smoothed)
        blue_channel_smoothed = mean_filter(blue_channel_smoothed)
        
        rgb_multiple_smoothed = np.stack((red_channel_smoothed, green_channel_smoothed, blue_channel_smoothed), axis=-1)
        
        r_equalized = equalize_channel(red_channel_smoothed)
        g_equalized = equalize_channel(green_channel_smoothed)
        b_equalized = equalize_channel(blue_channel_smoothed)
        
        rgb_equalized = np.stack((r_equalized, g_equalized, b_equalized), axis=-1)
        
        plt.imshow(rgb_single_smoothed)
        plt.title("RGB single smoothed Image")
        plt.show()
        
        plt.imshow(rgb_multiple_smoothed)
        plt.title("RGB multiple smoothed Image")
        plt.show()

        plt.imshow(rgb_equalized)
        plt.title("RGB equalized Image")
        plt.show()

        hsv_image = rgb2hsv(rgb_multiple_smoothed)
        plt.imshow(hsv_image)
        plt.title("HSV smoothed Image")
        plt.show()
        
        descriptor = method(grayscale_smoothed)
        descriptor_index = int(img_path.stem.split("_")[-1])
        descriptors[descriptor_index] = descriptor
        if save_as_pkl:
            with open(pkl_path, "wb") as f:
                pickle.dump(descriptor, f)
    return descriptors
