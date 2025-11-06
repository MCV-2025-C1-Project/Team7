import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from segmentation import get_mask_and_crops, hybrid_mask_fft_color_lab, get_mask_and_crops_refined#, grad_mag
import argparse


def process_folder_and_save(
    dataset_path: str,
    output_path: str,
    use_mask: str = "mask_lab",
    min_area: int = 1000,
    padding: int = 10,
):
    """
    Procesa todas las .jpg del dataset y guarda:
    - máscara seleccionada
    - visualización con bboxes
    - recortes de imagen y de máscara por componente
    - figura resumen (4 imágenes: RGB mask, FFT, Lab y detecciones)
    """
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(Path(dataset_path).glob("*.jpg"))
    if not img_paths:
        raise FileNotFoundError(f"No se encontraron imágenes en {dataset_path}")

    for img_path in img_paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"⚠️ No se pudo leer {img_path}, saltando...")
            continue

        # 1️⃣ Obtener todas las máscaras intermedias y recortes
        masks = hybrid_mask_fft_color_lab(img_bgr)
        res = get_mask_and_crops(
            [img_bgr],
            use_mask=use_mask,
            min_area=min_area,
            # padding=padding,
            reject_border=True,
            outermost_only=True,
        )

        # 2️⃣ Guardar máscara principal
        cv2.imwrite(str(out_dir / f"{img_path.stem}_{use_mask}.png"), res["mask"])

        # 3️⃣ Guardar visualización (bboxes)
        vis_bgr = cv2.cvtColor(res["vis"], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / f"{img_path.stem}_detected.png"), vis_bgr)

        # 4️⃣ Guardar recortes individuales
        for i, (crop_img, crop_mask) in enumerate(zip(res["crops_img"], res["crops_mask"])):
            cv2.imwrite(str(out_dir / f"{img_path.stem}_crop{i+1}.png"), crop_img)
            cv2.imwrite(str(out_dir / f"{img_path.stem}_crop{i+1}_mask.png"), crop_mask)

        # 5️⃣ Crear figura resumen (sin imagen original)
        titles = [
            "Segmentación RGB",
            "FFT Pasa-Altos",
            "Máscara Lab",
            "Detecciones (mask_lab)",
        ]
        images = [
            masks["mask_rgb"],
            masks["fft_edges"],
            masks["mask_lab"],
            res["vis"],
        ]

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs = axs.ravel()
        for i in range(4):
            axs[i].imshow(images[i], cmap="gray" if i < 3 else None)
            axs[i].set_title(titles[i])
            axs[i].axis("off")

        plt.tight_layout()
        fig_path = out_dir / f"{img_path.stem}_figure.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

        print(f"✅ {img_path.name}: {len(res['bboxes'])} recortes guardados")
        print(f"   Figura resumen: {fig_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Generar máscaras, detecciones y recortes")
    p.add_argument("--dataset", type=str, default="./datasets/qsdadri",
                   help="Carpeta con imágenes .jpg")
    p.add_argument("--output", type=str, default="./outputs_detected",
                   help="Carpeta de salida")
    p.add_argument("--use-mask", type=str, default="mask_lab",
                   choices=["mask_rgb", "fft_edges", "mask_lab", "combined", "refined"],
                   help="Cuál máscara usar para detección/recorte")
    p.add_argument("--min-area", type=int, default=1000,
                   help="Área mínima del componente")
    p.add_argument("--padding", type=int, default=10,
                   help="Padding extra alrededor de cada recorte")
    return p.parse_args()


def main():
    args = parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    process_folder_and_save(
        dataset_path=args.dataset,
        output_path=args.output,
        use_mask=args.use_mask,
        min_area=args.min_area,
        padding=args.padding,
    )

    print("Proceso completado.")


if __name__ == "__main__":
    main()
