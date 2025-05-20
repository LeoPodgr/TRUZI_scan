import os
from datetime import datetime
from ultralytics import YOLO

from preprocessing.preprocess import preprocess_dataset
from split import split_dataset


def generate_dataset_yaml(save_dir: str, class_names: list, yaml_path: str):
    train_path = os.path.abspath(os.path.join(save_dir, 'train', 'images'))
    val_path = os.path.abspath(os.path.join(save_dir, 'val', 'images'))

    content = f"""train: {train_path}
val: {val_path}
nc: {len(class_names)}
names: {class_names}
"""
    with open(yaml_path, "w") as f:
        f.write(content)

    print(f"[TRAINING] Сконфигурирован dataset.yaml по пути: {yaml_path}")
    print(f"[TRAINING] Содержимое dataset.yaml:\n{content}")


def training(config: dict):
    train_cfg = config

    print("[TRAINING] Этап 1: Предобработка...")
    preprocessed_path = preprocess_dataset(
        input_root=os.path.abspath(train_cfg["data_path"]),
        crop_box=train_cfg.get("crop_box", (40, 100, 1030, 808)),
        original_image_size=train_cfg.get("original_image_size", (1164, 873)),
        sam_checkpoint=train_cfg.get("sam_checkpoint", "sam_vit_l_0b3195.pth"),
        model_type=train_cfg.get("model_type", "vit_l"),
        device=train_cfg.get("device", "cuda"),
        do_crop=train_cfg.get("apply_crop", True),
        do_contrast=train_cfg.get("apply_contrast", True),
        do_sam=train_cfg.get("apply_sam", True),
        train=True
    )

    print("[TRAINING] Этап 2: Разбиение на train/val...")
    split_output = os.path.join(preprocessed_path, "YOLO")
    split_dataset(
        images_path=os.path.join(preprocessed_path, "images"),
        annotations_path=os.path.join(preprocessed_path, "labels"),
        output_path=split_output,
        train_ratio=1 - train_cfg.get("validation_split", 0.1)
    )

    print("[TRAINING] Этап 3: Генерация dataset.yaml...")
    yaml_path = os.path.abspath(os.path.join(split_output, "dataset.yaml"))
    class_names = train_cfg.get("class_names", ["cancer"])
    generate_dataset_yaml(split_output, class_names, yaml_path)

    print("[TRAINING] Этап 4: Запуск обучения...")
    pretrained = train_cfg.get("pretrained", False)
    model_path = "best.pt" if pretrained else "yolov8n.pt"
    model = YOLO(model_path)

    train_args = {
        "data": yaml_path,
        "epochs": train_cfg["epochs"],
        "batch": train_cfg["batch_size"],
        "imgsz": train_cfg["img_size"],
        "lr0": train_cfg["learning_rate"],
        "project": os.path.abspath(train_cfg["weights_save_path"]),
        "name": f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "exist_ok": True,
        "patience": train_cfg["early_stopping_patience"] if train_cfg.get("early_stopping") else 0
    }

    if train_cfg.get("augmentation", False):
        aug_cfg = train_cfg.get("augmentation_params", {})
        augmentations = {
            "flipud": aug_cfg.get("flip", 0.0),
            "fliplr": aug_cfg.get("flip", 0.0),
            "degrees": aug_cfg.get("rotation", 0.0) * 180,
            "brightness": aug_cfg.get("brightness", 0.0),
            "contrast": aug_cfg.get("contrast", 0.0),
            "perspective": aug_cfg.get("perspective", 0.0),
            "saturation": aug_cfg.get("saturation", 0.0),
            "hue": aug_cfg.get("hue", 0.0),
            "noise": aug_cfg.get("noise", 0.0),
            "cutout": aug_cfg.get("cutout", 0.0),
            "mosaic": aug_cfg.get("mosaic", 0.0),
        }
        train_args.update({k: v for k, v in augmentations.items() if v > 0.0})

    print("[TRAINING] Аргументы обучения:")
    for k, v in train_args.items():
        print(f"  {k}: {v}")

    model.train(**train_args)

    print(f"[TRAINING] Обучение завершено.\nВеса сохранены в: {train_args['project']}/{train_args['name']}")
