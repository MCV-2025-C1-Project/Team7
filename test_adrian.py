import cv2
import matplotlib.pyplot as plt
import os
from pathlib import Path

from segmentation import hybrid_mask_fft_color_lab  # tu función de máscara
from filtering import connected_components  # tu función de componentes conectados

# === CONFIGURACIÓN ===
DATASET_PATH = "./datasets/qsd2_w3"  # carpeta con tus imágenes
OUTPUT_PATH = "./outputs_detected"
os.makedirs(OUTPUT_PATH, exist_ok=True)


def main():
    # Obtener lista de imágenes
    img_paths = sorted(Path(DATASET_PATH).glob("*.jpg"))
    if len(img_paths) == 0:
        raise FileNotFoundError(f"No se encontraron imágenes en {DATASET_PATH}")

    for img_path in img_paths:
        print(f"Procesando: {img_path}")
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"⚠️ No se pudo leer {img_path}, saltando...")
            continue

        # Aplica el método híbrido
        masks = hybrid_mask_fft_color_lab(img_bgr)

        # Detectar cuadros (solo máscara y min_area)
        components = connected_components(masks["mask_lab"], min_area=1000)
        print(f"🖼️ {len(components)} cuadros detectados")

        # Visualización de máscaras intermedias
        titles = [
            "Segmentación RGB",
            "FFT Pasa-Altos",
            "Máscara Lab + Otsu",
            "Combinada",
            "Final Refinada",
        ]
        images = [
            masks["mask_rgb"],
            masks["fft_edges"],
            masks["mask_lab"],
            masks["combined"],
            masks["refined"],
        ]

        fig, axs = plt.subplots(2, 3, figsize=(16, 8))
        axs = axs.ravel()
        axs[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Imagen Original")
        for i in range(5):
            axs[i + 1].imshow(images[i], cmap="gray")
            axs[i + 1].set_title(titles[i])
        for ax in axs:
            ax.axis("off")
        plt.tight_layout()
        plt.show()

        # Mostrar detecciones
        vis = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).copy()
        for i, comp in enumerate(components):
            x1, y1, x2, y2 = comp["bbox"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                vis,
                f"#{i + 1}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        plt.figure(figsize=(8, 6))
        plt.imshow(vis)
        plt.title("Cuadros detectados")
        plt.axis("off")
        plt.show()

        fig_path = Path(OUTPUT_PATH) / f"{img_path.stem}_detected_figure.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
        plt.show()
        plt.close(fig)  # opcional, libera memoria en bucles largos

        img_path_out = Path(OUTPUT_PATH) / f"{img_path.stem}_detected.png"
        cv2.imwrite(str(img_path_out), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"Imagen guardada en {img_path_out}")
        print(f"Figura guardada en {fig_path}")

        # Guardar imagen con detecciones
        output_file = Path(OUTPUT_PATH) / f"{img_path.stem}_detected.png"
        cv2.imwrite(str(output_file), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"Imagen guardada en {output_file}\n")


if __name__ == "__main__":
    main()
