import os
import shutil
from preprocessing.crop import crop_directory_with_labels
from preprocessing.crop import crop_directory_without_labels
from preprocessing.contrast import enhance_contrast_in_directory
from preprocessing.sam import apply_sam_to_directory

def preprocess_dataset(input_root: str,
                       crop_box=(40, 100, 1030, 808),
                       original_image_size=(1164, 873),
                       sam_checkpoint="sam_vit_l_0b3195.pth",
                       model_type="vit_l",
                       device="cuda",
                       do_crop=True,
                       do_contrast=True,
                       do_sam=True,
                       train=False) -> str:

    copied_root = input_root.rstrip('/\\') + "_preproc"

    if os.path.exists(copied_root):
        print(f"[PREPROCESS] Удаляем существующую копию: {copied_root}")
        shutil.rmtree(copied_root)

    print(f"[PREPROCESS] Копируем входную папку: {input_root} → {copied_root}")
    shutil.copytree(input_root, copied_root)

    current_root = copied_root

    if do_crop:
        print("[PREPROCESS] Шаг 1: кроп изображений и разметки...")
        if train:
            crop_directory_with_labels(
                input_root=current_root,
                crop_y1=crop_box[1],
                crop_y2=crop_box[3],
                crop_x1=crop_box[0],
                crop_x2=crop_box[2]
            )
        else:
            crop_directory_without_labels(
                input_root=current_root,
                crop_y1=crop_box[1],
                crop_y2=crop_box[3],
                crop_x1=crop_box[0],
                crop_x2=crop_box[2]
            )

    if do_contrast:
        print("[PREPROCESS] Шаг 2: усиление контраста...")
        enhance_contrast_in_directory(
            input_root=os.path.join(current_root, "images")
        )

    if do_sam:
        print("[PREPROCESS] Шаг 3: SAM сегментация...")
        apply_sam_to_directory(
            input_root=current_root,
            sam_checkpoint=sam_checkpoint,
            model_type=model_type,
            device=device
        )

    print(f"[PREPROCESS] Предобработка завершена. Результат: {current_root}")
    return current_root
