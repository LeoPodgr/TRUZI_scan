import os
import cv2
import numpy as np
from typing import Tuple

def enhance_contrast(image_path: str,
                     output_path: str = None,
                     clip_limit: float = 2.0,
                     tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Изображение не найдено: {image_path}")

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    if output_path:
        cv2.imwrite(output_path, result)

    return result

def enhance_contrast_in_directory(input_root: str,
                                  clip_limit: float = 2.0,
                                  tile_grid_size: Tuple[int, int] = (8, 8),
                                  extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")) -> str:
    
    output_root = input_root.rstrip("/\\") + "_enhanced"
    os.makedirs(output_root, exist_ok=True)

    print(f"[INFO] Контрастная обработка изображений в {input_root}")
    for filename in os.listdir(input_root):
        if filename.lower().endswith(extensions):
            input_path = os.path.join(input_root, filename)
            output_path = os.path.join(output_root, filename)
            try:
                enhance_contrast(input_path, output_path, clip_limit, tile_grid_size)
            except Exception as e:
                print(f"[ERROR] Ошибка при обработке {filename}: {e}")

    print(f"[INFO] Обработка завершена: {output_root}")
    return output_root
