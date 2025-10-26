import cv2
import numpy as np
from filtering import laplacian_filter
from tqdm import tqdm
from denoise import denoise_image


def preprocess_images(
    images: dict[str, list[np.ndarray]], do_resize: bool = True, do_denoise: bool = True
) -> dict[str, list[np.ndarray]]:
    for img_name, img in images.items():
        processed_list = []
        for single_img in img:
            # a) Resize coherent
            if do_resize:
                single_img = cv2.resize(
                    single_img, (256, 256), interpolation=cv2.INTER_AREA
                )

            if do_denoise:
                # b) Gray-world white balance (balanceig simple de canals)
                img_f = single_img.astype(np.float32) + 1e-6
                mB, mG, mR = [img_f[:, :, c].mean() for c in range(3)]
                g = (mB + mG + mR) / 3.0
                img_f[:, :, 0] *= g / mB
                img_f[:, :, 1] *= g / mG
                img_f[:, :, 2] *= g / mR
                img_bal = np.clip(img_f, 0, 255).astype(np.uint8)

                # d) Suavitzat bilateral (treu soroll preservant vores i color)
                img_smooth = denoise_image(img_bal)
                # img_smooth = cv2.medianBlur(img_bal, 3)

                processed_list.append(img_smooth)
            else:
                processed_list.append(single_img)
        images[img_name] = processed_list
    return images


def preprocess_images_laplacian(images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    for img_name, img in tqdm(images.items(), desc="Preprocessing Images"):
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
        img_smooth = cv2.blur(img_bal, (5, 5))

        gray = cv2.cvtColor(img_smooth, cv2.COLOR_BGR2GRAY)
        lap_edges = laplacian_filter(gray)
        images[img_name] = lap_edges
    return images


def preprocess_images_for_segmentation(img):
    # 1. Convertir a escala de grisos o HSV
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Reduir soroll
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Millorar contrast
    equalized = cv2.equalizeHist(blur)

    return equalized
