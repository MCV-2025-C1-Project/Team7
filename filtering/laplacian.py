import numpy as np

def laplacian_filter(image: np.ndarray) -> np.ndarray:
    """
    Aplica un filtro Laplaciano manual para detectar bordes en una imagen en escala de grises.
    Implementación sin usar convolve2d.
    Args:
        image: Imagen 2D (grayscale) como un array numpy.
    Returns:
        Imagen filtrada (bordes resaltados) como un array numpy.
    """

    kernel = np.array([
        [0,  1,  0],
        [1, -4,  1],
        [0,  1,  0]
    ])

    image = image.astype(np.float32)

    padded = np.pad(image, pad_width=1, mode='reflect')

    output = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+3, j:j+3]
            output[i, j] = np.sum(region * kernel)

    output = np.clip(output, 0, 255).astype(np.uint8)

    return output
