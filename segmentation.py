import cv2
import numpy as np

from filtering import (
    closing,
    connected_components_cv2,
    dilate_mask,
    erode_mask,
    get_center_and_hollow_mask,
    segment_rgb_double_threshold,
)


def compute_binary_mask_1(img: np.ndarray) -> np.ndarray:
    """
    Computes a binary mask for the background and foreground
    Args:
        img: A np array representing the image from where to extract the mask
    Returns:
        A 2D uint8 np.array binary mask
    """

    # --- Segmentación ---
    mask = segment_rgb_double_threshold(img, low_thresh=(50, 50, 50), high_thresh=(240, 240, 240))

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

    delta_ab = np.sqrt((img_lab[:, :, 1] - wall_lab[1]) ** 2 + (img_lab[:, :, 2] - wall_lab[2]) ** 2).astype(np.float32)

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
    mask_rgb = segment_rgb_double_threshold(img_rgb, low_thresh=(50, 50, 50), high_thresh=(240, 240, 240))
    mask_rgb = cv2.morphologyEx(mask_rgb, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask_rgb = to_u8(mask_rgb)

    # 2) FFT -> región
    fft_edges = fft_highpass(gray, radius=25)
    edge_region = edges_to_region(fft_edges, thresh=35, dilate_ks=5)

    # 3) Lab + Otsu (todos los cuadros)
    mask_lab = compute_mask_lab_otsu(img_bgr)
    mask_lab = closing(mask_lab, k=15)
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

    delta_ab = np.sqrt((img_lab[:, :, 1] - wall_lab[1]) ** 2 + (img_lab[:, :, 2] - wall_lab[2]) ** 2).astype(np.float32)

    dist = cv2.normalize(delta_ab, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # limpieza SUAVE: no nos quedamos con el mayor componente
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (open_ks, open_ks)),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (close_ks, close_ks)),
    )
    mask = fill_holes(mask)
    return to_u8(mask)


"""
def fill_holes(mask: np.ndarray) -> np.ndarray:
    # Rellena agujeros en una máscara binaria 0/255 usando flood fill
    h, w = mask.shape
    flood = mask.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(mask, flood_inv)
    return filled
"""


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
    cv2.floodFill(tmp, lab, (0, 0), 128)  # marcar fondo conectado al borde
    mask_no_border = np.where(tmp == 128, 0, mask).astype(np.uint8)
    return mask_no_border


"""
def edges_to_region(edges: np.ndarray, thresh=40, dilate_ks=7) -> np.ndarray:
    # De bordes a región sólida
    binm = (edges > thresh).astype(np.uint8) * 255
    if dilate_ks > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_ks, dilate_ks))
    binm = cv2.dilate(binm, k, 1)
    binm = fill_holes(binm)
    binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return binm
"""


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
    cv2.floodFill(ff, lab, (0, 0), 255)  # fondo conectado al borde
    inv = cv2.bitwise_not(ff)
    return cv2.bitwise_or(mask, inv)


"""
def edges_to_region(edges, thresh=35, dilate_ks=5):
    binm = ((edges > thresh).astype(np.uint8)) * 255
    if dilate_ks > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_ks, dilate_ks))
    binm = cv2.dilate(binm, k, 1)
    binm = fill_holes(binm)
    binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return to_u8(binm)
"""


def edges_to_region(edges, thresh=35, dilate_ks=5):
    # 1) Umbral -> 0/1
    binm = (edges.astype(np.uint8) > thresh).astype(np.uint8)

    # 2) Dilatación opcional (sobre 0/1)
    if dilate_ks and dilate_ks > 0:
        # k = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_ks, dilate_ks))
        binm = (binm * 255).astype(np.uint8)
        if dilate_ks and dilate_ks > 0:
            binm = dilate_mask(binm, kernel_size=dilate_ks)
        binm = (binm > 0).astype(np.uint8)
    # 3) Pasamos a 0/255 para las operaciones OpenCV siguientes
    binm = (binm * 255).astype(np.uint8)

    # 4) Limpieza opcional (si no las necesitas, puedes comentar estas dos líneas)
    binm = fill_holes(binm)
    binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # 5) Forzar salida binaria exacta 0/255
    binm = ((binm > 0).astype(np.uint8)) * 255
    return binm


def get_crops_from_gt_mask(
    img_list: list[np.ndarray],
    mask_gt: np.ndarray,
    min_area: int = 2000,
    reject_border: bool = False,
    border_margin: int = 2,
    outermost_only: bool = True,
    padding: int = 10,
) -> list[np.ndarray]:
    """
    Generates crops from a ground truth mask (optimized with OpenCV).

    Args:
        img_list: List containing the input image in BGR format
        mask_gt: Ground truth mask (grayscale, 0=background, 255=object)
        min_area: Minimum area for a component to be kept
        reject_border: Whether to reject components touching the border
        border_margin: Margin for border rejection
        outermost_only: Whether to keep only outermost contours
        padding: Extra padding for crops

    Returns:
        List of cropped images sorted by x-coordinate (left to right)
    """
    img_bgr = img_list[0]
    H, W = img_bgr.shape[:2]

    # Ensure mask is binary (0/255)
    mask_binary = (mask_gt > 127).astype(np.uint8) * 255

    # Use OpenCV's connected components with stats (much faster)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_binary, connectivity=8, ltype=cv2.CV_32S
    )

    # Collect valid components
    valid_crops = []

    # Start from 1 to skip background (label 0)
    for label_id in range(1, num_labels):
        # Get component statistics
        x, y, w, h, area = stats[label_id]

        # Filter by minimum area
        if area < min_area:
            continue

        # Check if touching border
        if reject_border:
            if x <= border_margin or y <= border_margin or x + w >= W - border_margin or y + h >= H - border_margin:
                continue

        # Apply padding and clamp to image boundaries
        x1p = max(0, x - padding)
        y1p = max(0, y - padding)
        x2p = min(W - 1, x + w + padding)
        y2p = min(H - 1, y + h + padding)

        # Extract crop
        crop_img = img_bgr[y1p : y2p + 1, x1p : x2p + 1]

        if W > H:
            valid_crops.append((x1p, crop_img))
        else:
            valid_crops.append((y1p, crop_img))

    valid_crops.sort(key=lambda item: (item[0]))

    # Extract only the crops, limit to 2 if needed
    crops_img = [crop for _, crop in valid_crops]

    if len(crops_img) > 2:
        crops_img = crops_img[:2]

    return crops_img


# def get_mask_and_crops(
#     img_list: list[np.ndarray],
#     use_mask: str = "mask_lab",  # "mask_lab", "refined", "combined", etc.
#     min_area: int = 2000,
#     reject_border: bool = True,
#     border_margin: int = 2,
#     outermost_only: bool = True,
#     padding: int = 0,  # padding extra para los recortes
# ):
#     """
#     Devuelve máscara seleccionada y recortes por componente.
#     Returns:
#     {
#     "mask": np.ndarray (0/255),
#     "components": List[dict],
#     "crops_img": List[np.ndarray],
#     "crops_mask": List[np.ndarray],
#     "bboxes": List[tuple],
#     "vis": np.ndarray (RGB con detecciones dibujadas)
#     }
#     """
#     img_bgr = img_list[0]
#     # Generar máscaras
#     masks = hybrid_mask_fft_color_lab(img_bgr)
#     if use_mask not in masks:
#         raise KeyError(f"use_mask '{use_mask}' no está en {list(masks.keys())}")
#     mask_sel = masks[use_mask]

#     # Componentes
#     comps = connected_components_cv2(
#         mask_sel,
#         min_area=min_area,
#         reject_border=reject_border,
#         border_margin=border_margin,
#         outermost_only=outermost_only,
#     )

#     # Recortes
#     H, W = mask_sel.shape
#     crops_img, crops_mask, bboxes = [], [], []
#     for comp in comps:
#         x1, y1, x2, y2 = comp["bbox"]

#         # aplicar padding y acotar
#         x1p = max(0, x1 - padding)
#         y1p = max(0, y1 - padding)
#         x2p = min(W - 1, x2 + padding)
#         y2p = min(H - 1, y2 + padding)

#         crop_img = img_bgr[y1p : y2p + 1, x1p : x2p + 1]
#         crop_mask = mask_sel[y1p : y2p + 1, x1p : x2p + 1]

#         crops_img.append(crop_img)
#         crops_mask.append(crop_mask)
#         bboxes.append((x1p, y1p, x2p, y2p))
#     if len(crops_img) > 2:
#         crops_img = crops_img[:2]
#         crops_mask = crops_mask[:2]
#         bboxes = bboxes[:2]
#         decorated = [[int(bbox[0]), crop] for bbox, crop in zip(bboxes, crops_img)]
#         decorated.sort(key=lambda x: x[0])
#         crops_img = [crop for _, crop in decorated]

#     # Visualización
#     vis = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).copy()
#     for i, (x1, y1, x2, y2) in enumerate(bboxes):
#         cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(
#             vis,
#             f"#{i + 1}",
#             (x1, max(0, y1 - 10)),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.6,
#             (0, 255, 0),
#             2,
#         )

#     return {
#         "mask": mask_sel,
#         "components": comps,
#         "crops_img": crops_img,
#         "crops_mask": crops_mask,
#         "bboxes": bboxes,
#         "vis": vis,
#     }


def get_mask_and_crops(
    img_list: list[np.ndarray],
    use_mask: str = "mask_lab",
    min_area: int = 2000,
    reject_border: bool = True,
    border_margin: int = 2,
    outermost_only: bool = True,
):
    """
    Versión mejorada: intenta approximar el contorno a un cuadrilátero (p. ej. trapecio)
    usando convexHull + approxPolyDP con búsqueda de epsilon. Si falla, hace fallback
    a minAreaRect.
    """

    img_bgr = img_list[0]

    # Obtener máscaras
    masks = hybrid_mask_fft_color_lab(img_bgr)
    if use_mask not in masks:
        raise KeyError(f"use_mask '{use_mask}' no está en {list(masks.keys())}")
    mask_sel = masks[use_mask].astype(np.uint8)

    # Componentes conectados
    comps = connected_components_cv2(
        mask_sel,
        min_area=min_area,
        reject_border=reject_border,
        border_margin=border_margin,
        outermost_only=outermost_only,
    )

    H, W = mask_sel.shape
    crops_img, crops_mask, bboxes = [], [], []

    # Nueva máscara para rellenar los cuadriláteros mínimos
    refined_mask = np.zeros_like(mask_sel, dtype=np.uint8)

    for comp in comps:
        x1, y1, x2, y2 = comp["bbox"]

        # Recorte sin padding
        comp_mask = mask_sel[y1 : y2 + 1, x1 : x2 + 1]
        if np.count_nonzero(comp_mask) == 0:
            continue

        # findContours espera una imagen binaria; damos la máscara relativa al bbox
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # Elegimos el contorno más grande (por área)
        contour = max(contours, key=cv2.contourArea)

        # 1) Intentar con convexHull (el hull suaviza concavidades)
        hull = cv2.convexHull(contour)

        # 2) perímetro para guiar epsilon
        perim = cv2.arcLength(hull, True)

        # 3) búsqueda iterativa de epsilon para approxPolyDP intentando obtener 4 vértices
        found_quad = None
        # rango de eps: 0.5% .. 20% del perímetro, paso 0.5% (ajustable)
        eps_start = 0.005 * perim
        eps_end = 0.20 * perim
        eps_step = 0.005 * perim
        eps = eps_start
        while eps <= eps_end:
            approx = cv2.approxPolyDP(hull, eps, True)
            if approx is not None and len(approx) == 4:
                found_quad = approx
                break
            eps += eps_step

        # 4) si no hay resultado con hull, probar directamente sobre el contorno
        if found_quad is None:
            eps = eps_start
            while eps <= eps_end:
                approx = cv2.approxPolyDP(contour, eps, True)
                if approx is not None and len(approx) == 4:
                    found_quad = approx
                    break
                eps += eps_step

        # 5) fallback: si aún no hay quad, usar minAreaRect (rectángulo rotado)
        if found_quad is None:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)  # 4 puntos del rectángulo rotado
            box = np.int32(box)
        else:
            # approx viene en coordenadas relativas al recorte; reestructurar
            box = found_quad.reshape(4, 2).astype(np.int32)

        # Ajustar coordenadas al sistema original (sumar offset del bbox)
        box[:, 0] += x1
        box[:, 1] += y1

        # Dibujar el cuadrilátero en la nueva máscara
        cv2.fillPoly(refined_mask, [box], 255)

        # Guardar recorte original (sin padding)
        crop_img = img_bgr[y1 : y2 + 1, x1 : x2 + 1]
        crop_mask = mask_sel[y1 : y2 + 1, x1 : x2 + 1]

        if W > H:
            crops_img.append((x1, crop_img))
        else:
            crops_img.append((y1, crop_img))
        crops_mask.append(crop_mask)
        bboxes.append((x1, y1, x2, y2))

    crops_img.sort(key=lambda item: (item[0]))
    crops_img = [crop for _, crop in crops_img]
    if len(crops_img) > 2:
        crops_img = crops_img[:2]
    # Visualización
    vis = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).copy()
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"#{i+1}", (x1, max(0, y1-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    return {
        "mask": refined_mask,  # Nueva máscara con cuadriláteros/trapecios mínimos
        "components": comps,
        "crops_img": crops_img,
        "crops_mask": crops_mask,
        "bboxes": bboxes,
        "vis": vis,
    }


import numpy as np


def get_mask_and_crops_refined(
    img_list: list[np.ndarray],
    use_mask: str = "mask_lab",
    min_area: int = 2000,
    reject_border: bool = True,
    border_margin: int = 2,
    outermost_only: bool = True,
):
    """
    Versión mejorada: detecta cuadriláteros (incluyendo trapecios) y recalcula
    todos los componentes, crops, bboxes y visualización basados en las nuevas máscaras.
    """

    img_bgr = img_list[0]

    # Obtener máscaras originales
    masks = hybrid_mask_fft_color_lab(img_bgr)
    if use_mask not in masks:
        raise KeyError(f"use_mask '{use_mask}' no está en {list(masks.keys())}")
    mask_sel = masks[use_mask].astype(np.uint8)

    # Obtener componentes iniciales (para recortar)
    comps = connected_components_cv2(
        mask_sel,
        min_area=min_area,
        reject_border=reject_border,
        border_margin=border_margin,
        outermost_only=outermost_only,
    )

    # Máscara refinada donde dibujaremos cuadriláteros
    refined_mask = np.zeros_like(mask_sel, dtype=np.uint8)

    for comp in comps:
        x1, y1, x2, y2 = comp["bbox"]

        comp_mask = mask_sel[y1 : y2 + 1, x1 : x2 + 1]
        if np.count_nonzero(comp_mask) == 0:
            continue

        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)

        # --- 1️⃣ Buscar cuadrilátero (trapecio) ---
        hull = cv2.convexHull(contour)
        perim = cv2.arcLength(hull, True)

        found_quad = None
        eps_start, eps_end, eps_step = 0.005 * perim, 0.20 * perim, 0.005 * perim
        eps = eps_start

        while eps <= eps_end:
            approx = cv2.approxPolyDP(hull, eps, True)
            if approx is not None and len(approx) == 4:
                found_quad = approx
                break
            eps += eps_step

        if found_quad is None:
            eps = eps_start
            while eps <= eps_end:
                approx = cv2.approxPolyDP(contour, eps, True)
                if approx is not None and len(approx) == 4:
                    found_quad = approx
                    break
                eps += eps_step

        if found_quad is None:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
        else:
            box = found_quad.reshape(4, 2).astype(np.int32)

        # --- 2️⃣ Ajustar coordenadas y dibujar en la máscara global ---
        box[:, 0] += x1
        box[:, 1] += y1
        cv2.fillPoly(refined_mask, [box], 255)

    # --- 3️⃣ Recalcular componentes sobre la máscara refinada ---
    refined_comps = connected_components_cv2(
        refined_mask,
        min_area=min_area,
        reject_border=reject_border,
        border_margin=border_margin,
        outermost_only=outermost_only,
    )

    # --- 4️⃣ Generar crops, máscaras y bboxes basadas en la nueva máscara ---
    
    H, W = mask_sel.shape
    crops_img, crops_mask, bboxes = [], [], []
    for comp in refined_comps:
        x1, y1, x2, y2 = comp["bbox"]

        crop_img = img_bgr[y1:y2+1, x1:x2+1]
        crop_mask = refined_mask[y1:y2+1, x1:x2+1]

        # crops_img.append(crop_img)
        if W > H:
            crops_img.append((x1, crop_img))
        else:
            crops_img.append((y1, crop_img))
        crops_mask.append(crop_mask)
        bboxes.append((x1, y1, x2, y2))
    crops_img.sort(key=lambda item: (item[0]))
    crops_img = [crop for _, crop in crops_img]
    if len(crops_img) > 2:
        crops_img = crops_img[:2]
    # --- 5️⃣ Visualización final basada en componentes refinados ---
    vis = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).copy()
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"#{i+1}", (x1, max(0, y1-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    return {
        "mask": refined_mask,          # Nueva máscara con cuadriláteros/trapecios
        "components": refined_comps,   # Componentes recalculados
        "crops_img": crops_img,        # Imágenes recortadas nuevas
        "crops_mask": crops_mask,      # Máscaras recortadas nuevas
        "bboxes": bboxes,              # Bboxes recalculadas
        "vis": vis,                    # Visualización final
    }
