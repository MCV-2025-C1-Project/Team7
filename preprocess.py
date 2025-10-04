import cv2
import numpy as np


def preprocess_images(images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    for img_name, img in images.items():
        # a) Resize coherent
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

        # b) Gray-world white balance (balanceig simple de canals)
        img_f = img.astype(np.float32) + 1e-6
        mB, mG, mR = [img_f[:, :, c].mean() for c in range(3)]
        g = (mB + mG + mR) / 3.0
        img_f[:, :, 0] *= g / mB
        img_f[:, :, 1] *= g / mG
        img_f[:, :, 2] *= g / mR
        img_bal = np.clip(img_f, 0, 255).astype(np.uint8)

        # c) CLAHE sobre el canal V (millora contrast/brillantor de forma estable)
        hsv = cv2.cvtColor(img_bal, cv2.COLOR_BGR2HSV)
        V = hsv[:, :, 2]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        hsv[:, :, 2] = clahe.apply(V)
        img_clahe = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # d) Suavitzat bilateral (treu soroll preservant vores i color)
        img_smooth = cv2.bilateralFilter(img_clahe, d=7, sigmaColor=60, sigmaSpace=7)

        images[img_name] = cv2.cvtColor(
            img_smooth, cv2.COLOR_BGR2RGB
        )  # <-- Transformar a RGB per la funció manual del rgb histogram
    return images
