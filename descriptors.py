from collections.abc import Callable

import numpy as np
from pathlib import Path
import pickle

import cv2
from skimage.feature import local_binary_pattern
from scipy.fftpack import dctn


def grayscale_histogram(image: np.ndarray, bins: int = 256) -> np.ndarray:
    """
    Compute the grayscale histogram of an image.
    Args:
        image: A 2D numpy array representing a grayscale image.
    Returns:
        A 1D numpy array of length 256 representing the histogram.
    """

    histogram = np.zeros(bins, dtype=int)
    bin_size = 256 // bins
    for pixel in image.flatten():
        histogram[pixel // bin_size] += 1
    return histogram


def rgb_hist_hellinger(image: np.ndarray, bins: int = 8) -> np.ndarray:
    """
    Compute the RGB histogram of an image using Hellinger bins.
    Args:
        image: A 3D numpy array representing an RGB image.
        bins: Number of bins per channel.
    Returns:
        A 1D numpy array of length bins*3 representing the concatenated histogram.
    """

    hist = np.zeros((bins, 3), dtype=np.float32)

    bin_size = 256 // bins
    for r, g, b in image.reshape(-1, 3):
        hist[r // bin_size, 0] += 1
        hist[g // bin_size, 1] += 1
        hist[b // bin_size, 2] += 1

    # Apply Hellinger normalization
    hist = np.sqrt(hist / (hist.sum() + 1e-8))

    return hist.flatten().astype(np.float32)


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


def rgb_histogram(image: np.ndarray, bins=8) -> np.ndarray:
    """
    Compute the 3D RGB histogram -> Can't use for week 1.
    Args:
        image: A 3D numpy array representing an RGB image.
    Returns:
        A 1D numpy array of length bins^3 representing the 3D histogram.
    """
    # Ensure uint8 input
    img = image.astype(np.uint8)

    # Compute bin indices for each channel
    bin_edges = np.linspace(0, 256, bins + 1, endpoint=True)
    r_idx = np.digitize(img[:, :, 2].ravel(), bin_edges) - 1
    g_idx = np.digitize(img[:, :, 1].ravel(), bin_edges) - 1
    b_idx = np.digitize(img[:, :, 0].ravel(), bin_edges) - 1

    flat_idx = b_idx * (bins * bins) + g_idx * bins + r_idx

    hist = np.bincount(flat_idx, minlength=bins**3).astype(np.float32)

    return hist


def hsv_histogram(image: np.ndarray, bins=[16, 16, 8]) -> np.ndarray:
    """
    Compute the 3D HSV histogram.
    Args:
        image: A 3D numpy array representing an BGR image.
    Returns:
        A 1D numpy array of length bins[0]*bins[1]*bins[2] representing the 3D histogram.
    """
    assert image.shape[2] == 3, "Input image must have 3 channels (BGR format)."
    assert len(bins) == 3, "Bins must be a list of three integers."

    img_hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2HSV)

    # Scale values to bin indices directly
    h_scale = bins[0] / 256.0
    s_scale = bins[1] / 256.0
    v_scale = bins[2] / 256.0

    h_idx = (img_hsv[:, :, 0] * h_scale).astype(np.int32)
    s_idx = (img_hsv[:, :, 1] * s_scale).astype(np.int32)
    v_idx = (img_hsv[:, :, 2] * v_scale).astype(np.int32)

    # Clamp to valid range
    h_idx = np.clip(h_idx, 0, bins[0] - 1).ravel()
    s_idx = np.clip(s_idx, 0, bins[1] - 1).ravel()
    v_idx = np.clip(v_idx, 0, bins[2] - 1).ravel()

    # Compute linear indices
    flat_idx = h_idx * (bins[1] * bins[2]) + s_idx * bins[2] + v_idx

    # Count occurrences
    hist = np.bincount(flat_idx, minlength=bins[0] * bins[1] * bins[2]).astype(
        np.float32
    )

    return hist / (hist.sum() + 1e-8)  # Normalize histogram


def hsv_block_histogram_concat(
    image: np.ndarray, bins: list[int], grid: tuple[int, int]
) -> np.ndarray:
    """
    Compute the concatenated HSV histograms of image blocks. Uses the function hsv_histogram.
    Args:
        image: A 3D numpy array representing an BGR image.
        bins: Number of bins per channel.
        grid: Tuple representing the number of blocks in (rows, cols).
    Returns:
        A 1D numpy array representing the concatenated histograms of all blocks.
    """
    assert image.shape[2] == 3, "Input image must have 3 channels (BGR format)."
    assert len(bins) == 3, "Bins must be a list of three integers."
    assert len(grid) == 2, "Grid must be a tuple of two integers (rows, cols)."

    # Compute height&width block size and initialize list for histograms
    h_block, w_block = image.shape[0] // grid[0], image.shape[1] // grid[1]

    # preallocate output array for efficiency
    hist_size = bins[0] * bins[1] * bins[2]
    total_blocks = grid[0] * grid[1]
    histograms = np.empty(total_blocks * hist_size, dtype=np.float32)

    block_idx = 0
    for i in range(grid[0]):
        for j in range(grid[1]):
            # get image block
            block = image[
                i * h_block : (i + 1) * h_block, j * w_block : (j + 1) * w_block
            ]
            # compute histogram for block and append to list
            hist = hsv_histogram(block, bins)
            histograms[block_idx * hist_size : (block_idx + 1) * hist_size] = hist
            block_idx += 1

    return histograms


def hsv_block_hist_concat_func(
    bins: list[int] = [16, 16, 8], grid: tuple[int, int] = (2, 2)
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Wrapper function that returns the function hsv_block_histogram_concat but with the desired bins and grid parameters set.
    Args:
        bins: Number of bins per channel.
        grid: Tuple representing the number of blocks in (rows, cols).
    Returns:
        A callable function that takes an image and returns the concatenated histogram, with the specified bins and grid.
    """

    def custom_hsv_block_hist_concat(img_bgr: np.ndarray) -> np.ndarray:
        return hsv_block_histogram_concat(img_bgr, bins, grid)

    return custom_hsv_block_hist_concat


def hsv_hierarchical_block_histogram_concat(
    img_bgr: np.ndarray, bins: list[int], levels_grid: list[tuple[int, int]]
) -> np.ndarray:
    """
    Compute the concatenated HSV histograms of hierarchical image blocks. Uses the function hsv_block_histogram_concat.
    Args:
        img_bgr: A 3D numpy array representing an BGR image.
        bins: Number of bins per channel.
        levels_grid: List of tuples representing the number of blocks in (rows, cols) for each level.
    Returns:
        A 1D numpy array representing the concatenated histograms of all blocks across levels.
    """
    assert img_bgr.shape[2] == 3, "Input image must have 3 channels (BGR format)."
    assert len(bins) == 3, "Bins must be a list of three integers."
    assert all(len(grid) == 2 for grid in levels_grid), (
        "Each grid must be a tuple of two integers (rows, cols)."
    )

    # preallocate output array for efficiency
    hist_size = bins[0] * bins[1] * bins[2]
    total_blocks = sum(grid[0] * grid[1] for grid in levels_grid)
    histograms = np.empty(total_blocks * hist_size, dtype=np.float32)

    block_idx = 0
    for grid in levels_grid:
        hist = hsv_block_histogram_concat(img_bgr, bins, grid)
        num_blocks = grid[0] * grid[1]
        histograms[block_idx * hist_size : (block_idx + num_blocks) * hist_size] = hist
        block_idx += num_blocks

    return histograms


def hsv_hier_block_hist_concat_func(
    bins: list[int] = [16, 16, 8],
    levels_grid: list[tuple[int, int]] = [(1, 1), (2, 2), (3, 3)],
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Wrapper function that returns the function hsv_hierarchical_block_histogram_concat but with the desired bins and levels_grid parameters set.
    Args:
        bins: Number of bins per channel.
        levels_grid: List of tuples representing the number of blocks in (rows, cols) for each level.
    Returns:
        A callable function that takes an image and returns the concatenated histogram, with the specified bins and levels_grid.
    """

    def custom_hsv_hier_block_hist_concat(img_bgr: np.ndarray) -> np.ndarray:
        return hsv_hierarchical_block_histogram_concat(img_bgr, bins, levels_grid)

    return custom_hsv_hier_block_hist_concat


def hsv_histogram_concat(img_bgr: np.ndarray, bins=[16, 16, 8]) -> np.ndarray:
    """
    Compute a 1D concatenated HSV histogram
    Args:
        img_bgr: A 3D numpy array representing an BGR image.

    Returns:
        A 1D numpy array representing the concatenated histogram.
    """
    img_hsv = cv2.cvtColor(img_bgr.astype(np.uint8), cv2.COLOR_BGR2HSV)
    h_bins = bins[0]
    s_bins = bins[1]
    v_bins = bins[2]

    # --- H channel ---
    H = img_hsv[:, :, 0].ravel()
    H_bin_edges = np.linspace(0, 256, h_bins + 1)
    H_idx = np.digitize(H, H_bin_edges) - 1
    H_hist = np.zeros(h_bins, dtype=np.float32)
    for idx in H_idx:
        H_hist[idx] += 1

    # --- S channel ---
    S = img_hsv[:, :, 1].ravel()
    S_bin_edges = np.linspace(0, 256, s_bins + 1)
    S_idx = np.digitize(S, S_bin_edges) - 1
    S_hist = np.zeros(s_bins, dtype=np.float32)
    for idx in S_idx:
        S_hist[idx] += 1

    # --- V channel ---
    V = img_hsv[:, :, 2].ravel()
    V_bin_edges = np.linspace(0, 256, v_bins + 1)
    V_idx = np.digitize(V, V_bin_edges) - 1
    V_hist = np.zeros(v_bins, dtype=np.float32)
    for idx in V_idx:
        V_hist[idx] += 1

    hist = np.concatenate([H_hist, S_hist, V_hist]).astype(np.float32)
    return hist / (hist.sum() + 1e-8)  # Normalize histogram


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
    cs = cs.astype("uint8")

    # get the value from cumulative sum for every index in flat, and set that as img_new
    img_new = cs[img.flatten()]

    # put array back into original shape since we flattened it
    img_new = np.reshape(img_new, img.shape)

    return img_new


def lbp_descriptor_histogram(
    image: np.ndarray, lbp_p: int = 8, lbp_r: int = 1, bins: int = 256
) -> np.ndarray:
    """
    Function that computes the LBP histogram of an image. First converts the image to grayscale, then computes the LBP and finally the histogram.

    Args:
        image (np.ndarray): Input image in BGR format.

    Returns:
        np.ndarray: 1D numpy array representing the LBP histogram with 256 bins.
    """

    assert image.shape[2] == 3, "Input image must have 3 channels (BGR format)."

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Compute LBP
    lbp = local_binary_pattern(gray, P=lbp_p, R=lbp_r, method="uniform")
    # Compute histogram
    lbp_hist, _ = np.histogram(lbp, bins=bins, density=True)
    return lbp_hist


def lbp_descriptor_histogram_func(
    lbp_p: int = 8, lbp_r: int = 1, bins: int = 256
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Wrapper function that returns the function lbp_descriptor_histogram but with the desired lbp_p, lbp_r and bins parameters set.

    Args:
        lbp_p (int): Number of circularly symmetric neighbour set points (quantization of the angular space).
        lbp_r (int): Radius of circle (spatial resolution of the operator).
        bins (int): Number of bins for the histogram.

    Returns:
        Callable[[np.ndarray], np.ndarray]: A callable function that takes an image and returns the LBP histogram, with the specified parameters.
    """

    def custom_lbp_descriptor_histogram(img_bgr: np.ndarray) -> np.ndarray:
        return lbp_descriptor_histogram(img_bgr, lbp_p, lbp_r, bins)

    return custom_lbp_descriptor_histogram


def dct_descriptor(
    image: np.ndarray, block_size: int = 8, n_coefs: int = 9
) -> np.ndarray:
    """
    Function that computes the DCT descriptor of an image. First converts the image to grayscale, then computes the DCT and finally returns the top-left coefficients flattened.

    Args:
        image (np.ndarray): Input image in BGR format.
        n_coefs (int): Number of coefficients to keep.
    Returns:
        np.ndarray: 1D numpy array representing the DCT descriptor.
    """

    assert image.shape[2] == 3, "Input image must have 3 channels (BGR format)."

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Compute DCT
    dct = dctn(gray, shape=(block_size, block_size), norm="ortho")
    zigzag_indices = [
        (i, j)
        for s in range(2 * block_size - 1)
        for i in range(max(0, s - block_size + 1), min(s + 1, block_size))
        for j in [s - i]
    ]
    dct_descriptor = np.array([dct[i, j] for i, j in zigzag_indices[:n_coefs]])
    return dct_descriptor


def dct_block_descriptor(
    image: np.ndarray,
    grid: tuple[int, int] = (2, 2),
    n_coefs: int = 9,
    relative_coefs: bool = False,
) -> np.ndarray:
    """
    Computes the DCT descriptor for each block in the image. Uses the function dct_descriptor.

    Args:
        image (np.ndarray): Input image in BGR format.
        grid (tuple[int, int], optional): Number of blocks in the vertical and horizontal directions. Defaults to (2, 2).

    Returns:
        np.ndarray: 2D numpy array representing the DCT descriptors for each block.
    """
    assert image.shape[2] == 3, "Input image must have 3 channels (BGR format)."
    assert len(grid) == 2, "Grid must be a tuple of two integers (rows, cols)."
    assert n_coefs > 0, "Number of coefficients must be positive."

    h, w = image.shape[:2]
    block_h = h // grid[0]
    block_w = w // grid[1]
    if relative_coefs:
        n_coefs = int(block_h * block_w * n_coefs / 100)  # percentage of total coefs
    dct_descriptors = np.empty((grid[0] * grid[1] * n_coefs), dtype=np.float32)

    for i in range(grid[0]):
        for j in range(grid[1]):
            block = image[
                i * block_h : (i + 1) * block_h, j * block_w : (j + 1) * block_w
            ]
            dct_desc = dct_descriptor(block, block_size=block_h, n_coefs=n_coefs)
            dct_descriptors[
                (i * grid[1] + j) * n_coefs : (i * grid[1] + j + 1) * n_coefs
            ] = dct_desc

    return np.array(dct_descriptors)


def dct_block_descriptor_func(
    grid: tuple[int, int] = (2, 2), n_coefs: int = 9, relative_coefs: bool = False
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Wrapper function that returns the function dct_block_descriptor but with the desired grid and n_coefs parameters set.

    Args:
        grid (tuple[int, int], optional): Number of blocks in the vertical and horizontal directions. Defaults to (2, 2).
        n_coefs (int, optional): Number of coefficients to keep. Defaults to 9.

    Returns:
        Callable[[np.ndarray], np.ndarray]: A callable function that takes an image and returns the DCT block descriptor, with the specified parameters.
    """

    def custom_dct_block_descriptor(img_bgr: np.ndarray) -> np.ndarray:
        return dct_block_descriptor(img_bgr, grid, n_coefs, relative_coefs)

    return custom_dct_block_descriptor


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
    # Determine the path for the .pkl file based on parameters
    pkl_path = (
        Path(__file__).parent
        / f"{suffix}-{method.__name__}-{('grayscale' if use_grayscale else 'color')}.pkl"
    )

    # Load descriptors from .pkl if it exists and overwrite is False
    if pkl_path.exists() and not overwrite_pkl:
        with open(pkl_path, "rb") as f:
            descriptors = pickle.load(f)
    else:
        descriptors = {}
        # for each image, compute the descriptor and store it in the dictionary
        for (
            imgname,
            image,
        ) in images.items():  # tqdm.tqdm(images.items(), desc="Computing descriptors"):
            descriptor = method(image)
            descriptor_index = int(imgname.split("_")[-1])
            descriptors[descriptor_index] = descriptor
        if save_as_pkl:
            with open(pkl_path, "wb") as f:
                pickle.dump(descriptors, f)
    return descriptors
