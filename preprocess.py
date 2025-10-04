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

        # d) Suavitzat bilateral (treu soroll preservant vores i color)
        img_smooth = cv2.blur(img_bal, (5,5))

        images[img_name] = img_smooth
    return images
