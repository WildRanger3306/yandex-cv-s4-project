import os
import json
import random
import re
from pathlib import Path
import shutil
from PIL import Image

# Пути
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
VQA_LABEL_FILE = PROCESSED_DIR / "vqa_labels_t4.json" # Используем полный результат с Colab

# Выходной датасет для YOLO
YOLO_ROOT = Path("data/yolo_dataset")
YOLO_ROOT.mkdir(parents=True, exist_ok=True)

(YOLO_ROOT / "images/train").mkdir(parents=True, exist_ok=True)
(YOLO_ROOT / "images/val").mkdir(parents=True, exist_ok=True)
(YOLO_ROOT / "labels/train").mkdir(parents=True, exist_ok=True)
(YOLO_ROOT / "labels/val").mkdir(parents=True, exist_ok=True)

# Имена классов (у нас только один класс - человек без шлема)
CLASSES = ["no_helmet"]

def parse_boxes(text):
    """
    Пытается извлечь координаты из ответа модели, даже если там есть рассуждения.
    Ожидаемый формат: [...] внутри ```json или просто как список.
    """
    try:
        # 1. Сначала ищем блок ```json ... ```
        json_match = re.search(r'```json\s*(\[.*?\])\s*```', text, re.DOTALL)
        # 2. Если нет, ищем просто любую структуру [ ... ]
        if not json_match:
            json_match = re.search(r'(\[.*\])', text, re.DOTALL)
            
        if json_match:
            data = json.loads(json_match.group(1))
            boxes = []
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'bbox_2d' in item:
                        boxes.append(item['bbox_2d'])
                    elif isinstance(item, list) and len(item) == 4:
                        boxes.append(item)
            return boxes
    except Exception as e:
        pass
        
    # Fallback на регулярки для поиска всех [y1, x1, y2, x2] в тексте
    found_boxes = re.findall(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', text)
    return [[int(x) for x in box] for box in found_boxes]

def convert_to_yolo(box, img_w, img_h):
    """
    ПОДТВЕРЖДЕНО ЭКСПЕРИМЕНТОМ:
    VLM возвращает [xmin, ymin, xmax, ymax] в ПИКСЕЛЯХ.
    """
    xmin, ymin, xmax, ymax = box
    
    # Ограничиваем координаты границами изображения
    xmin = max(0, min(xmin, img_w))
    xmax = max(0, min(xmax, img_w))
    ymin = max(0, min(ymin, img_h))
    ymax = max(0, min(ymax, img_h))

    # YOLO формат: x_center, y_center, width, height (нормализованные 0-1)
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    x_center = (xmin + xmax) / (2 * img_w)
    y_center = (ymin + ymax) / (2 * img_h)
    
    return x_center, y_center, w, h

def main():
    if not VQA_LABEL_FILE.exists():
        print(f"Error: Label file {VQA_LABEL_FILE} not found! Run Phase 3 first.")
        return

    with open(VQA_LABEL_FILE, "r") as f:
        vqa_results = json.load(f)

    all_images = list(vqa_results.keys())
    random.shuffle(all_images)
    
    split_idx = int(len(all_images) * 0.8)
    train_imgs = all_images[:split_idx]
    val_imgs = all_images[split_idx:]
    
    print(f"Processing dataset: {len(train_imgs)} train, {len(val_imgs)} val")

    for subset, img_list in [("train", train_imgs), ("val", val_imgs)]:
        for img_name in img_list:
            # 1. Копируем картинку
            src_path = RAW_DATA_DIR / img_name
            if not src_path.exists():
                # Пробуем найти рекурсивно (если распаковалось во вложенные папки)
                src_path = next(RAW_DATA_DIR.rglob(img_name), None)
            
            if not src_path:
                continue
                
            # 1. Получаем размеры картинки
            with Image.open(src_path) as img:
                img_w, img_h = img.size
                
            dest_img = YOLO_ROOT / "images" / subset / img_name
            shutil.copy(src_path, dest_img)
            
            # 2. Создаем файл разметки
            label_text = vqa_results[img_name]
            boxes = parse_boxes(label_text)
            
            label_file = YOLO_ROOT / "labels" / subset / (Path(img_name).stem + ".txt")
            
            with open(label_file, "w") as f:
                for box in boxes:
                    # Нормализуем по РЕАЛЬНОМУ размеру изображения
                    yolo_box = convert_to_yolo(box, img_w, img_h)
                    # Формат: class_id x y w h
                    f.write(f"0 {' '.join([f'{c:.6f}' for c in yolo_box])}\n")

    # Создаем data.yaml для YOLOv8
    yaml_content = f"""
path: {YOLO_ROOT.absolute()}
train: images/train
val: images/val

names:
  0: {CLASSES[0]}
"""
    with open(YOLO_ROOT / "data.yaml", "w") as f:
        f.write(yaml_content)

    print(f"Dataset ready at {YOLO_ROOT}")
    print(f"Generated data.yaml at {YOLO_ROOT / 'data.yaml'}")

if __name__ == "__main__":
    main()
