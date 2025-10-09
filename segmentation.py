from filtering import (
        segment_rgb_double_threshold,
        segment_rgb_threshold,
        erode_mask,
        dilate_mask,
        compute_centroid,
        get_center_and_crop,
        get_center_and_mask_crop,
        get_center_and_hollow_mask
)

import numpy as np

def compute_binary_mask(img: np.ndarray) -> np.ndarray:
    """
    Computes a binary mask for the background and foreground
    Args:
        img: A np array representing the image from where to extract the mask
    Returns:
        A 2D uint8 np.array binary mask
    """
    
    # --- Segmentación ---
    mask = segment_rgb_double_threshold(img, low_thresh=(50,50,50), high_thresh=(240,240,240))

    # Invertir máscara si el fondo es blanco
    mask = 255 - mask
    
    # --- Morfología vectorizada ---
    mask_clean = dilate_mask(mask, kernel_size=11)
    mask_clean = erode_mask(mask_clean, kernel_size=11)
    
    _, _, mask_filled = get_center_and_hollow_mask(mask_clean, padding=0)
    
    return mask_filled