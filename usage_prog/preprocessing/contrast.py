import os
import cv2
import numpy as np
from typing import Tuple

def enhance_contrast(image_path: str,
                     clip_limit: float = 2.0,
                     tile_grid_size: Tuple[int, int] = (8, 8)) -> None:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Изображение не найдено: {image_path}")

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    cv2.imwrite(image_path, result)

def enhance_contrast_in_directory(input_root: str,
                                  clip_limit: float = 2.0,
                                  tile_grid_size: Tuple[int, int] = (8, 8),
                                  extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")) -> str:
    print(f"[INFO] Контрастная обработка изображений в: {input_root}")
    for filename in os.listdir(input_root):
        if filename.lower().endswith(extensions):
            image_path = os.path.join(input_root, filename)
            try:
                enhance_contrast(image_path, clip_limit, tile_grid_size)
            except Exception as e:
                print(f"[ERROR] Ошибка при обработке {filename}: {e}")
    print(f"[INFO] Контрастная обработка завершена.")
    return input_root
