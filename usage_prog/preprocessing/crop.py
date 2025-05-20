import os
import cv2

def pre_crop(image, y1, y2, x1, x2):
    return image[y1:y2, x1:x2]

def adjust_yolo_bbox(cls, x, y, w, h, img_w, img_h, crop_left, crop_top, crop_width, crop_height):
    abs_x = x * img_w
    abs_y = y * img_h
    abs_w = w * img_w
    abs_h = h * img_h

    x1 = abs_x - abs_w / 2
    y1 = abs_y - abs_h / 2
    x2 = abs_x + abs_w / 2
    y2 = abs_y + abs_h / 2

    x1 -= crop_left
    x2 -= crop_left
    y1 -= crop_top
    y2 -= crop_top

    if x2 <= 0 or y2 <= 0 or x1 >= crop_width or y1 >= crop_height:
        return None

    x1 = max(0, min(x1, crop_width))
    x2 = max(0, min(x2, crop_width))
    y1 = max(0, min(y1, crop_height))
    y2 = max(0, min(y2, crop_height))

    new_w = x2 - x1
    new_h = y2 - y1
    new_x = x1 + new_w / 2
    new_y = y1 + new_h / 2

    return f"{cls} {new_x / crop_width:.6f} {new_y / crop_height:.6f} {new_w / crop_width:.6f} {new_h / crop_height:.6f}"

def crop_directory_with_labels(input_root: str,
                                crop_y1: int, crop_y2: int,
                                crop_x1: int, crop_x2: int) -> None:
    images_dir = os.path.join(input_root, "images")
    labels_dir = os.path.join(input_root, "labels")

    crop_width = crop_x2 - crop_x1
    crop_height = crop_y2 - crop_y1

    for filename in os.listdir(images_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(images_dir, filename)
            img = cv2.imread(image_path)

            if img is None:
                print(f"[WARNING] Не удалось загрузить {filename}")
                continue

            img_h, img_w = img.shape[:2]
            img_crop = pre_crop(img, crop_y1, crop_y2, crop_x1, crop_x2)
            cv2.imwrite(image_path, img_crop)  

            label_file = os.path.splitext(filename)[0] + ".txt"
            label_path = os.path.join(labels_dir, label_file)

            if not os.path.exists(label_path):
                continue

            with open(label_path, 'r') as infile:
                lines = infile.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                cls, x, y, w, h = parts
                adjusted = adjust_yolo_bbox(cls, float(x), float(y), float(w), float(h),
                                            img_w, img_h,
                                            crop_x1, crop_y1,
                                            crop_width, crop_height)
                if adjusted:
                    new_lines.append(adjusted)

            with open(label_path, 'w') as outfile:
                outfile.write('\n'.join(new_lines))

def crop_directory_without_labels(input_root: str,
                                  crop_y1: int, crop_y2: int,
                                  crop_x1: int, crop_x2: int) -> None:
    images_dir = os.path.join(input_root, "images")

    for filename in os.listdir(images_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(images_dir, filename)
            img = cv2.imread(image_path)

            if img is None:
                print(f"[WARNING] Не удалось загрузить {filename}")
                continue

            img_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
            cv2.imwrite(image_path, img_crop)
