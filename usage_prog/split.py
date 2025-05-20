import os
import random
import shutil
from collections import defaultdict
from typing import Tuple

def split_dataset(
    images_path: str,
    annotations_path: str,
    output_path: str,
    train_ratio: float = 0.8,
    postfix_delimiter: str = "_",
    seed: int = 42
) -> Tuple[str, str]:
    
    random.seed(seed)

    def extract_postfix(filename):
        return filename.split(postfix_delimiter)[-1]

    images = [f for f in os.listdir(images_path) if f.endswith((".jpg", ".png"))]
    groups = defaultdict(list)
    for image in images:
        postfix = extract_postfix(image)
        groups[postfix].append(image)

    group_keys = list(groups.keys())
    random.shuffle(group_keys)

    train_count = int(len(group_keys) * train_ratio)
    train_groups = group_keys[:train_count]
    val_groups = group_keys[train_count:]

    train_images = set(image for group in train_groups for image in groups[group])
    val_images = set(image for group in val_groups for image in groups[group])

    intersection = train_images & val_images
    if intersection:
        raise ValueError(f"Обнаружены дублирующиеся файлы между train и val: {intersection}")

    def move_files(image_list, dest_images_path, dest_labels_path):
        os.makedirs(dest_images_path, exist_ok=True)
        os.makedirs(dest_labels_path, exist_ok=True)

        for image_name in image_list:
            image_src = os.path.join(images_path, image_name)
            label_name = image_name.rsplit(".", 1)[0] + ".txt"
            label_src = os.path.join(annotations_path, label_name)

            shutil.copy(image_src, os.path.join(dest_images_path, image_name))
            if os.path.exists(label_src):
                shutil.copy(label_src, os.path.join(dest_labels_path, label_name))

    train_images_dir = os.path.join(output_path, "train", "images")
    train_labels_dir = os.path.join(output_path, "train", "labels")
    val_images_dir = os.path.join(output_path, "val", "images")
    val_labels_dir = os.path.join(output_path, "val", "labels")

    move_files(train_images, train_images_dir, train_labels_dir)
    move_files(val_images, val_images_dir, val_labels_dir)

    print("[SPLIT] Разделение завершено!")
    return os.path.join(output_path, "train"), os.path.join(output_path, "val")
