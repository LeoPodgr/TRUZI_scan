import json
import sys
from train import training
from inference import inference

def main():
    try:
        with open("conf.json", "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        print(f"[ERROR] Ошибка загрузки config.json: {e}")
        sys.exit(1)

    mode = config.get("mode")

    if mode == "training":
        training(config["training"])
    elif mode == "inference":
        inference(config["inference"])
    else:
        print(f"[ERROR] Неверный режим: {mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
