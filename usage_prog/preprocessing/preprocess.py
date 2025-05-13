import os
from crop import crop_images_and_labels
from contrast import enhance_contrast_in_directory
from sam import apply_sam_to_directory

def preprocess_dataset(input_root: str,
                       crop_box=(40, 100, 1030, 808),
                       original_image_size=(1164, 873),
                       sam_checkpoint="sam_vit_l_0b3195.pth",
                       model_type="vit_l",
                       device="cuda",
                       do_crop=True,
                       do_contrast=True,
                       do_sam=True) -> str:

    current_root = input_root

    if do_crop:
        print("[PREPROCESS] Шаг 1: кроп изображений и разметки...")
        current_root = crop_images_and_labels(
            input_root=current_root,
            crop_box=crop_box,
            original_image_size=original_image_size
        )

    if do_contrast:
        print("[PREPROCESS] Шаг 2: усиление контраста...")
        current_root = enhance_contrast_in_directory(
            input_dir=os.path.join(current_root, "images")
        )

    if do_sam:
        print("[PREPROCESS] Шаг 3: SAM сегментация...")
        current_root = apply_sam_to_directory(
            input_root=current_root,
            sam_checkpoint=sam_checkpoint,
            model_type=model_type,
            device=device
        )
        current_root = os.path.join(current_root, "images_with_contours")

    print(f"[PREPROCESS] Предобработка завершена. Результат: {current_root}")
    return current_root
