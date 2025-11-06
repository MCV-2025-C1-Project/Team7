from typing import Callable
import numpy as np
import cv2

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


# ADVO INI : Laplacian Filter
def laplacian_filter(image: np.ndarray) -> np.ndarray:
    """
    Aplica un filtro Laplaciano manual para detectar bordes en una imagen en escala de grises.
    Implementación sin usar convolve2d.
    Args:
        image: Imagen 2D (grayscale) como un array numpy.
    Returns:
        Imagen filtrada (bordes resaltados) como un array numpy.
    """

    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    image = image.astype(np.float32)

    padded = np.pad(image, pad_width=1, mode="reflect")

    output = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i : i + 3, j : j + 3]
            output[i, j] = np.sum(region * kernel)

    output = np.clip(output, 0, 255).astype(np.uint8)

    return output


def segment_rgb_double_threshold(
    image_rgb, low_thresh=(80, 80, 80), high_thresh=(230, 230, 230)
):
    """
    Segmenta un cuadro en RGB usando un rango por canal.
    """
    mask = (
        (image_rgb[:, :, 0] >= low_thresh[0])
        & (image_rgb[:, :, 0] <= high_thresh[0])
        & (image_rgb[:, :, 1] >= low_thresh[1])
        & (image_rgb[:, :, 1] <= high_thresh[1])
        & (image_rgb[:, :, 2] >= low_thresh[2])
        & (image_rgb[:, :, 2] <= high_thresh[2])
    )
    return mask.astype(np.uint8) * 255


def segment_rgb_threshold(image_rgb, threshold=200):
    """
    Segmenta un cuadro en RGB usando un único threshold global.
    """
    mask = (
        (image_rgb[:, :, 0] < threshold)
        & (image_rgb[:, :, 1] < threshold)
        & (image_rgb[:, :, 2] < threshold)
    )
    return mask.astype(np.uint8) * 255


def erode_mask(mask, kernel_size=3):
    """
    Erosiona una máscara binaria de manera vectorizada.
    """
    pad = kernel_size // 2
    padded = np.pad(mask, pad, mode="constant", constant_values=0)
    H, W = mask.shape
    shape = (H, W, kernel_size, kernel_size)
    strides = padded.strides * 2
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    eroded = np.all(windows == 255, axis=(2, 3)).astype(np.uint8) * 255
    return eroded


def dilate_mask(mask, kernel_size=3):
    """
    Dilata una máscara binaria de manera vectorizada.
    """
    pad = kernel_size // 2
    padded = np.pad(mask, pad, mode="constant", constant_values=0)
    H, W = mask.shape
    shape = (H, W, kernel_size, kernel_size)
    strides = padded.strides * 2
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    dilated = np.any(windows == 255, axis=(2, 3)).astype(np.uint8) * 255
    return dilated


def opening(mask: np.ndarray, k: int = 3) -> np.ndarray:
    return dilate_mask(erode_mask(mask, kernel_size=k), kernel_size=k)


def closing(mask: np.ndarray, k: int = 3) -> np.ndarray:
    return erode_mask(dilate_mask(mask, kernel_size=k), kernel_size=k)


def compute_centroid(mask):
    """
    Calcula el centroide de la máscara binaria.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    cx = int(np.mean(xs))
    cy = int(np.mean(ys))
    return (cx, cy)


def get_center_and_crop(mask, img, padding=10):
    """
    Calcula el centroide, bounding box y recorte dinámico de un objeto en una máscara binaria.

    Args:
        mask (np.ndarray): máscara binaria del objeto (0 o 255)
        img (np.ndarray): imagen original en BGR o RGB
        padding (int): píxeles extra alrededor del bounding box

    Returns:
        center (tuple): (cx, cy) del centroide
        bbox (tuple): (x_min, y_min, x_max, y_max) del bounding box
        crop (np.ndarray): recorte de la imagen original
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None, None, None  # no se encontró objeto

    # Bounding box
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # Centroide
    cx = int((x_min + x_max) / 2)
    cy = int((y_min + y_max) / 2)
    center = (cx, cy)

    # Recorte dinámico con padding
    x1 = max(0, x_min - padding)
    x2 = min(img.shape[1], x_max + padding)
    y1 = max(0, y_min - padding)
    y2 = min(img.shape[0], y_max + padding)
    crop = img[y1:y2, x1:x2]

    bbox = (x_min, y_min, x_max, y_max)
    return center, bbox, crop


def get_center_and_mask_crop(mask, padding=10):
    """
    Calcula el centroide, bounding box y recorte de la máscara binaria.
    No recorta la imagen, solo devuelve la máscara filtrada.

    Args:
        mask (np.ndarray): máscara binaria del objeto (0 o 255)
        padding (int): píxeles extra alrededor del bounding box

    Returns:
        center (tuple): (cx, cy) del centroide relativo al recorte
        bbox (tuple): (x_min, y_min, x_max, y_max) en coordenadas originales
        mask_crop (np.ndarray): máscara recortada al bounding box con padding
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None, None, None  # no hay objeto

    # Bounding box
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # Aplicar padding
    x1 = max(0, x_min - padding)
    x2 = min(mask.shape[1], x_max + padding)
    y1 = max(0, y_min - padding)
    y2 = min(mask.shape[0], y_max + padding)

    # Recorte de la máscara
    mask_crop = mask[y1:y2, x1:x2]

    # Centroide relativo al recorte
    ys_crop, xs_crop = np.where(mask_crop > 0)
    cx = int(xs_crop.mean())
    cy = int(ys_crop.mean())
    center = (cx, cy)

    bbox = (x1, y1, x2, y2)
    return center, bbox, mask_crop


def get_center_and_hollow_mask(mask, padding=10):
    """
    Calcula el centroide, bounding box y máscara hueca (0 dentro del objeto, 1 fuera).

    Args:
        mask (np.ndarray): máscara binaria del objeto (0 o 255)
        padding (int): píxeles extra alrededor del bounding box

    Returns:
        center (tuple): (cx, cy) del centroide del objeto original
        bbox (tuple): (x_min, y_min, x_max, y_max) del bounding box con padding
        mask_hollow (np.ndarray): máscara transformada (0 dentro del bounding box, 1 fuera)
    """
    ys, xs = np.where(mask > 0)
    mask_hollow = np.zeros_like(mask, dtype=np.uint8)  # por defecto todo fuera = 0

    if len(xs) == 0 or len(ys) == 0:
        return None, None, mask_hollow  # no hay objeto

    # Bounding box con padding
    x_min, x_max = max(0, xs.min() - padding), min(mask.shape[1], xs.max() + padding)
    y_min, y_max = max(0, ys.min() - padding), min(mask.shape[0], ys.max() + padding)

    # Hueco en la máscara
    mask_hollow[y_min:y_max, x_min:x_max] = 1

    # Centroide del objeto original (no del hueco)
    cx = int(xs.mean())
    cy = int(ys.mean())
    center = (cx, cy)

    bbox = (x_min, y_min, x_max, y_max)
    return center, bbox, mask_hollow


"""
def connected_components(mask, min_area=500, connectivity=8):
    
    Detecta componentes conectados en una máscara binaria (0/255) usando NumPy puro.
    Sin bucles anidados sobre todos los píxeles, solo sobre los blancos.

    Args:
        mask (np.ndarray): máscara binaria
        min_area (int): área mínima del componente
        connectivity (int): 4 u 8 (conectividad)

    Returns:
        List[dict]: cada elemento contiene:
            - 'bbox': (x_min, y_min, x_max, y_max)
            - 'center': (cx, cy)
            - 'area': número de píxeles

    # Convertimos a binario 0/1
    mask = (mask > 0).astype(np.uint8)
    H, W = mask.shape

    # Obtenemos coordenadas de píxeles blancos
    points = np.argwhere(mask)
    if len(points) == 0:
        return []

    # Creamos un set para lookup rápido
    points_set = set(map(tuple, points))
    components = []

    # Definimos vecinos
    if connectivity == 8:
        neighbors = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
    else:
        neighbors = [(-1, 0), (0, -1), (0, 1), (1, 0)]

    while points_set:
        # Tomamos un punto y hacemos BFS
        seed = points_set.pop()
        stack = [seed]
        comp_pixels = []

        while stack:
            y, x = stack.pop()
            comp_pixels.append((y, x))
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if (ny, nx) in points_set:
                    points_set.remove((ny, nx))
                    stack.append((ny, nx))

        comp_pixels = np.array(comp_pixels)
        area = len(comp_pixels)
        if area < min_area:
            continue

        ys, xs = comp_pixels[:, 0], comp_pixels[:, 1]
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        cx = int(xs.mean())
        cy = int(ys.mean())

        components.append(
            {"bbox": (x_min, y_min, x_max, y_max), "center": (cx, cy), "area": area}
        )

    return components
"""


def connected_components_cv2(
    mask,
    min_area=500,
    connectivity=8,
    reject_border=True,
    border_margin=0,
    outermost_only=True,
):
    """
    Detecta componentes conectados en una máscara binaria (0/255) usando OpenCV.
    Devuelve una lista de componentes con bbox, centro y área.
    """
    mask = (mask > 0).astype(np.uint8)
    H, W = mask.shape

    # Obtener etiquetas y estadísticas
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity)

    components = []
    for i in range(1, num_labels):  # ignorar fondo (label 0)
        x, y, w, h, area = stats[i]

        if area < min_area:
            continue

        x_min, y_min, x_max, y_max = x, y, x + w - 1, y + h - 1

        # Descartar si toca el borde
        if reject_border:
            if (
                x_min <= border_margin
                or y_min <= border_margin
                or x_max >= W - 1 - border_margin
                or y_max >= H - 1 - border_margin
            ):
                continue

        cx, cy = centroids[i]
        components.append(
            {
                "bbox": (x_min, y_min, x_max, y_max),
                "center": (int(cx), int(cy)),
                "area": int(area),
            }
        )

    # Eliminar componentes contenidos dentro de otros (outermost_only)
    if outermost_only and components:
        keep = [True] * len(components)

        def contained(bi, bj, tol=0):
            xi1, yi1, xi2, yi2 = bi
            xj1, yj1, xj2, yj2 = bj
            return (
                xi1 >= xj1 + tol
                and yi1 >= yj1 + tol
                and xi2 <= xj2 - tol
                and yi2 <= yj2 - tol
            )

        for i in range(len(components)):
            if not keep[i]:
                continue
            bi = components[i]["bbox"]
            for j in range(len(components)):
                if i == j or not keep[j]:
                    continue
                bj = components[j]["bbox"]
                if contained(bi, bj, tol=1):
                    keep[i] = False
                    break

        components = [c for k, c in enumerate(components) if keep[k]]

    components.sort(key=lambda c: c["area"], reverse=True)
    return components
def connected_components(
    mask,
    min_area=500,
    connectivity=8,
    reject_border=True,
    border_margin=0,
    outermost_only=True,
):
    """
    Detecta componentes conectados en una máscara binaria (0/255) usando NumPy puro.
    Args:
        mask (np.ndarray): máscara binaria
        min_area (int): área mínima del componente
        connectivity (int): 4 u 8
        reject_border (bool): si True, descarta componentes que toquen el borde
        border_margin (int): margen adicional para considerar "tocar borde"
        outermost_only (bool): si True, elimina componentes contenidos dentro de otros

    Returns:
        List[dict]: cada elemento contiene:
            - 'bbox': (x_min, y_min, x_max, y_max)
            - 'center': (cx, cy)
            - 'area': número de píxeles
    """
    mask = (mask > 0).astype(np.uint8)
    H, W = mask.shape

    points = np.argwhere(mask)
    if len(points) == 0:
        return []

    points_set = set(map(tuple, points))
    components = []

    if connectivity == 8:
        neighbors = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
    else:
        neighbors = [(-1, 0), (0, -1), (0, 1), (1, 0)]

    while points_set:
        seed = points_set.pop()
        stack = [seed]
        comp_pixels = []

        while stack:
            y, x = stack.pop()
            comp_pixels.append((y, x))
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if (ny, nx) in points_set:
                    points_set.remove((ny, nx))
                    stack.append((ny, nx))

        comp_pixels = np.array(comp_pixels)
        area = len(comp_pixels)
        if area < min_area:
            continue

        ys, xs = comp_pixels[:, 0], comp_pixels[:, 1]
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        # Opción: descartar componentes que toquen el borde
        if reject_border:
            if (
                x_min <= border_margin
                or y_min <= border_margin
                or x_max >= W - 1 - border_margin
                or y_max >= H - 1 - border_margin
            ):
                continue

        cx = int(xs.mean())
        cy = int(ys.mean())

        components.append(
            {"bbox": (x_min, y_min, x_max, y_max), "center": (cx, cy), "area": area}
        )

    # Opción: quedarnos con los más exteriores (eliminar los contenidos)
    if outermost_only and components:
        keep = [True] * len(components)

        def contained(bi, bj, tol=0):
            xi1, yi1, xi2, yi2 = bi
            xj1, yj1, xj2, yj2 = bj
            return (
                xi1 >= xj1 + tol
                and yi1 >= yj1 + tol
                and xi2 <= xj2 - tol
                and yi2 <= yj2 - tol
            )

        for i in range(len(components)):
            if not keep[i]:
                continue
            bi = components[i]["bbox"]
            for j in range(len(components)):
                if i == j or not keep[j]:
                    continue
                bj = components[j]["bbox"]
                # si el bbox i está contenido en j, descartamos i
                if contained(bi, bj, tol=1):
                    keep[i] = False
                    break

        components = [c for k, c in enumerate(components) if keep[k]]
    components.sort(key=lambda c: c["area"], reverse=True)
    return components



# ADVO FIN : Laplacian Filter
