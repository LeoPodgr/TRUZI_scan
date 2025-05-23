import os
import cv2
import numpy as np
from tqdm import tqdm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def draw_contour_overlay(masks, shape, thickness=2, border_dist=15):
    h, w = shape
    contour_layer = np.zeros((h, w), dtype=np.uint8)

    for mask in masks:
        binary_mask = np.zeros((h, w), dtype=np.uint8)
        binary_mask[mask['segmentation']] = 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cv2.drawContours(contour_layer, [cnt], -1, color=255, thickness=thickness)

    contour_layer[:border_dist, :] = 0
    contour_layer[-border_dist:, :] = 0
    contour_layer[:, :border_dist] = 0
    contour_layer[:, -border_dist:] = 0

    return contour_layer


def create_alpha_contour_overlay(image_gray, contour_mask, alpha_value=128):
    image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    h, w = contour_mask.shape
    overlay_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    overlay_rgba[:, :, :3] = 255
    overlay_rgba[:, :, 3] = 0
    overlay_rgba[contour_mask > 0] = [0, 0, 0, alpha_value]

    base_rgba = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2BGRA)
    result = base_rgba.copy()
    mask_alpha = overlay_rgba[:, :, 3] / 255.0
    for c in range(3):
        result[:, :, c] = result[:, :, c] * (1 - mask_alpha) + overlay_rgba[:, :, c] * mask_alpha

    return result[:, :, :3].astype(np.uint8)  # Вернуть только BGR


def process_image_with_sam(image_path, mask_generator, roi_box, contour_thickness, alpha_value, border_dist):
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    roi = image_rgb[roi_box[1]:roi_box[3], roi_box[0]:roi_box[2]]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    masks = mask_generator.generate(roi)
    h, w = roi.shape[:2]

    contour_mask = draw_contour_overlay(masks, (h, w), thickness=contour_thickness, border_dist=border_dist)
    final_roi_overlay = create_alpha_contour_overlay(roi_gray, contour_mask, alpha_value=alpha_value)

    image_bgr[roi_box[1]:roi_box[3], roi_box[0]:roi_box[2]] = final_roi_overlay
    return image_bgr


def apply_sam_to_directory(input_root: str,
                           sam_checkpoint: str = "sam_vit_l_0b3195.pth",
                           model_type: str = "vit_l",
                           device: str = "cuda",
                           roi_box=(200, 250, 770, 675),
                           contour_thickness=2,
                           alpha_value=128,
                           border_dist=15) -> str:
    images_dir = os.path.join(input_root, "images")
    print("[INFO] Инициализация SAM...")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    print(f"[INFO] SAM обработка изображений в {images_dir}")
    for filename in tqdm(os.listdir(images_dir)):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        image_path = os.path.join(images_dir, filename)
        try:
            result_img = process_image_with_sam(
                image_path, mask_generator, roi_box, contour_thickness, alpha_value, border_dist
            )
            cv2.imwrite(image_path, result_img)  # Перезаписываем изображение
        except Exception as e:
            print(f"[ERROR] Ошибка при обработке {filename}: {e}")

    print("[INFO] SAM обработка завершена.")
    return input_root
