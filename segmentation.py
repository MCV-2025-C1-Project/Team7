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
import cv2
import numpy as np

def compute_binary_mask_1(img: np.ndarray) -> np.ndarray:
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

def compute_binary_mask_2(img: np.ndarray):
    
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    h, w, _ = img_lab.shape

    border_width = 5
    border_mask = np.zeros((h, w), np.uint8)
    border_mask[:border_width, :] = 1
    border_mask[-border_width:, :] = 1
    border_mask[:, :border_width] = 1
    border_mask[:, -border_width:] = 1

    border_pixels = img_lab[border_mask == 1]
    wall_lab = np.median(border_pixels, axis=0).astype(np.uint8)

    delta_ab = np.sqrt(
        (img_lab[:, :, 1] - wall_lab[1]) ** 2 +
        (img_lab[:, :, 2] - wall_lab[2]) ** 2
    ).astype(np.float32)
        
    use_otsu = False
    if use_otsu:
        dist_uint8 = cv2.normalize(delta_ab, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, mask = cv2.threshold(dist_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        T = 10.0  # manual threshold in Lab space units
        mask = (delta_ab > T).astype(np.uint8) * 255
        
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    _, _, mask_filled = get_center_and_hollow_mask(mask, padding=0)

    return mask_filled