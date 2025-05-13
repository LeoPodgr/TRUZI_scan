import os
import cv2
from ultralytics import YOLO
from preprocessing import preprocess_dataset 
from datetime import datetime

def inference(config: dict):
    infer_cfg = config["inference"]
    train_cfg = config["training"]

    print("[INFERENCE] Этап 1: Предобработка входных данных...")
    preprocessed_path = preprocess_dataset(
        input_root=infer_cfg["image_path"],
        crop_box=train_cfg.get("crop_box", (40, 100, 1030, 808)),
        original_image_size=train_cfg.get("original_image_size", (1164, 873)),
        sam_checkpoint=train_cfg.get("sam_checkpoint", "sam_vit_l_0b3195.pth"),
        model_type=train_cfg.get("model_type", "vit_l"),
        device=train_cfg.get("device", "cuda"),
        apply_crop=train_cfg.get("apply_crop", True),
        apply_contrast=train_cfg.get("apply_contrast", True),
        apply_sam=train_cfg.get("apply_sam", True)
    )

    processed_images_path = os.path.join(preprocessed_path, "images")
    image_files = [
        os.path.join(processed_images_path, f)
        for f in os.listdir(processed_images_path)
        if f.endswith((".jpg", ".png"))
    ]
    if not image_files:
        raise FileNotFoundError(f"[INFERENCE] Нет изображений в: {processed_images_path}")

    print("[INFERENCE] Этап 2: Загрузка модели...")
    model = YOLO(infer_cfg["model_path"])

    conf_thres = infer_cfg.get("confidence_threshold", 0.5)
    iou_thres = infer_cfg.get("iou_threshold", 0.5)
    output_path = infer_cfg["output_path"]
    os.makedirs(output_path, exist_ok=True)

    for img_path in image_files:
        print(f"[INFERENCE] Обработка изображения: {img_path}")
        results = model.predict(
            source=img_path,
            conf=conf_thres,
            iou=iou_thres,
            save=infer_cfg.get("save_results", False),
            save_txt=infer_cfg.get("save_results", False),
            show=infer_cfg.get("show_results", False),
            project=output_path,
            name=f"infer_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            exist_ok=True
        )

        if infer_cfg.get("save_results", False) and not infer_cfg.get("show_results", False):
            annotated = results[0].plot()  
            base_name = os.path.basename(img_path)
            save_name = os.path.join(output_path, f"pred_{base_name}")
            cv2.imwrite(save_name, annotated)
            print(f"[INFERENCE] Сохранён результат: {save_name}")

    print("[INFERENCE] Готово.")
