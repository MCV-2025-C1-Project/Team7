from filtering import (
    segment_rgb_double_threshold,
    segment_rgb_threshold,
    erode_mask,
    dilate_mask,
    compute_centroid,
    get_center_and_crop,
    get_center_and_mask_crop,
    get_center_and_hollow_mask,
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
    mask = segment_rgb_double_threshold(
        img, low_thresh=(50, 50, 50), high_thresh=(240, 240, 240)
    )

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
        (img_lab[:, :, 1] - wall_lab[1]) ** 2 + (img_lab[:, :, 2] - wall_lab[2]) ** 2
    ).astype(np.float32)

    use_otsu = False
    if use_otsu:
        dist_uint8 = cv2.normalize(delta_ab, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        _, mask = cv2.threshold(dist_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        T = 10.0  # manual threshold in Lab space units
        mask = (delta_ab > T).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    _, _, mask_filled = get_center_and_hollow_mask(mask, padding=0)

    return mask_filled


# =====================================================
# === FFT PASA-ALTOS (rápido, sin convolución manual) ===
# =====================================================
def fft_highpass(img_gray, radius=30):
    """
    Aplica un filtro pasa-altos con FFT.
    """
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)

    rows, cols = img_gray.shape
    crow, ccol = rows // 2, cols // 2

    mask = np.ones((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), radius, 0, -1)

    fshift_filtered = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.abs(np.fft.ifft2(f_ishift))

    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img_back


# =====================================================
# === MÉTODO HÍBRIDO (LAB + FFT + COLOR + MORFOLOGÍA) ===
# =====================================================
def hybrid_mask_fft_color_lab(img_bgr):
    """
    Combina:
      1. Segmentación RGB (doble threshold)
      2. FFT (bordes)
      3. Tu compute_binary_mask_2 (Lab + Otsu)
      4. Morfología manual
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    # 1) RGB
    mask_rgb = segment_rgb_double_threshold(
        img_rgb, low_thresh=(70, 70, 70), high_thresh=(230, 230, 230)
    )
    mask_rgb = cv2.morphologyEx(mask_rgb, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask_rgb = to_u8(mask_rgb)

    # 2) FFT -> región
    fft_edges = fft_highpass(gray, radius=25)
    edge_region = edges_to_region(fft_edges, thresh=35, dilate_ks=5)

    # 3) Lab + Otsu (todos los cuadros)
    mask_lab = compute_mask_lab_otsu(img_bgr)

    # 4) Combinación robusta (equiv. a OR)
    combined = cv2.max(mask_lab, cv2.max(mask_rgb, edge_region))

    # 5) Refinado LIGERO (sin borrar objetos)
    refined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    refined = fill_holes(refined)
    refined = to_u8(refined)

    return {
        "mask_rgb": mask_rgb,
        "fft_edges": fft_edges,
        "mask_lab": mask_lab,
        "combined": combined,
        "refined": refined,
    }


# =====================================================
# === compute_binary_mask_2 con Otsu activado ===
# =====================================================
def compute_mask_lab_otsu(img_bgr, open_ks=3, close_ks=5):
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    h, w, _ = img_lab.shape
    bw = 5
    border = np.zeros((h, w), np.uint8)
    border[:bw, :] = border[-bw:, :] = border[:, :bw] = border[:, -bw:] = 1
    wall_lab = np.median(img_lab[border == 1], axis=0).astype(np.uint8)

    delta_ab = np.sqrt(
        (img_lab[:, :, 1] - wall_lab[1]) ** 2 +
        (img_lab[:, :, 2] - wall_lab[2]) ** 2
    ).astype(np.float32)

    dist = cv2.normalize(delta_ab, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # limpieza SUAVE: no nos quedamos con el mayor componente
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (open_ks, open_ks)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (close_ks, close_ks)))
    mask = fill_holes(mask)
    return to_u8(mask)
def fill_holes(mask: np.ndarray) -> np.ndarray:
    # Rellena agujeros en una máscara binaria 0/255 usando flood fill
    h, w = mask.shape
    flood = mask.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(mask, flood_inv)
    return filled

def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    if not cnts:
        return out
    c = max(cnts, key=cv2.contourArea)
    cv2.drawContours(out, [c], -1, 255, -1)
    return out

def clear_border(mask: np.ndarray) -> np.ndarray:
    # Elimina componentes que tocan el borde
    h, w = mask.shape
    lab = np.zeros((h + 2, w + 2), np.uint8)
    tmp = mask.copy()
    cv2.floodFill(tmp, lab, (0, 0), 128) # marcar fondo conectado al borde
    mask_no_border = np.where(tmp == 128, 0, mask).astype(np.uint8)
    return mask_no_border

def edges_to_region(edges: np.ndarray, thresh=40, dilate_ks=7) -> np.ndarray:
    # De bordes a región sólida
    binm = (edges > thresh).astype(np.uint8) * 255
    if dilate_ks > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_ks, dilate_ks))
    binm = cv2.dilate(binm, k, 1)
    binm = fill_holes(binm)
    binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return binm

def to_u8(mask):
    mask = mask.astype(np.uint8)
    if mask.max() <= 1:
        mask = mask * 255
    return mask

def fill_holes(mask):
    mask = to_u8(mask)
    h, w = mask.shape
    ff = mask.copy()
    lab = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, lab, (0, 0), 255) # fondo conectado al borde
    inv = cv2.bitwise_not(ff)
    return cv2.bitwise_or(mask, inv)

def edges_to_region(edges, thresh=35, dilate_ks=5):
    binm = ((edges > thresh).astype(np.uint8)) * 255
    if dilate_ks > 0:
       k = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_ks, dilate_ks))
    binm = cv2.dilate(binm, k, 1)
    binm = fill_holes(binm)
    binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return to_u8(binm)