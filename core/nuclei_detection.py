import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2


from histocartography.preprocessing import NucleiExtractor

def visualize_nuclei(image_path, output_dir):
    from openslide import OpenSlide

    # --- Load image (.svs or .png) ---
    if image_path.lower().endswith('.svs'):
        slide = OpenSlide(image_path)
        level = 0
        patch_size = 4096

        width, height = slide.level_dimensions[level]
        if width < patch_size or height < patch_size:
            raise ValueError(f"SVS slide {image_path} is too small to extract patch.")

        x = (width // 2) - (patch_size // 2)
        y = (height // 2) - (patch_size // 2)

        image = np.array(slide.read_region((x, y), level, (patch_size, patch_size)).convert("RGB"))
    else:
        image = np.array(Image.open(image_path).convert("RGB"))

    # --- Run nuclei detection ---
    nuclei_mask, centroids = NucleiExtractor().process(image)

   # --- Visualization ---
    os.makedirs(os.path.join(output_dir, "nuclei_detection"), exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(output_dir, "nuclei_detection", base_name + "_nuclei.png")

    # Convert nuclei mask to uint8 for OpenCV
    nuclei_mask_uint8 = nuclei_mask.astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(nuclei_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours directly on image using OpenCV
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, color=(0, 255, 0), thickness=3)

    # Save image using PIL (preserves exact output with no border)
    Image.fromarray(contour_image).save(save_path)


    print(f"[✓] Saved nuclei visualization to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input image (.png or .svs)")
    parser.add_argument("--output", type=str, required=True, help="Folder to save visualization")
    args = parser.parse_args()

    visualize_nuclei(args.input, args.output)
