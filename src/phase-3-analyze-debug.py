import json
import re
from pathlib import Path
from PIL import Image, ImageDraw

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
VQA_LABEL_FILE = PROCESSED_DIR / "vqa_labels_t4.json"
DEBUG_OUTPUT = PROCESSED_DIR / "debug_boxes.jpg"

def parse_boxes(text):
    try:
        clean_text = re.sub(r'```json|```', '', text).strip()
        data = json.loads(clean_text)
        return [item['bbox_2d'] for item in data if 'bbox_2d' in item]
    except:
        found = re.findall(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', text)
        return [[int(x) for x in b] for b in found]

def main():
    # Берем конкретную картинку для теста (из вашего JSON)
    img_name = "hard_hat_workers330.png"
    
    with open(VQA_LABEL_FILE, "r") as f:
        vqa_results = json.load(f)
    
    src_path = next(RAW_DATA_DIR.rglob(img_name), None)
    raw_box = parse_boxes(vqa_results[img_name])[0]
    
    img = Image.open(src_path).convert("RGB")
    w, h = img.size
    
    # Создаем холст для 3-х вариантов
    canvas = Image.new("RGB", (w * 3, h), "white")
    
    # Вариант 1: Текущий [ymin, xmin, ymax, xmax] -> Top, Left, Bottom, Right
    var1 = img.copy()
    draw1 = ImageDraw.Draw(var1)
    y1, x1, y2, x2 = raw_box
    draw1.rectangle([x1*w/1000, y1*h/1000, x2*w/1000, y2*h/1000], outline="red", width=5)
    
    # Вариант 2: Перевернутый [xmin, ymin, xmax, ymax] -> Left, Top, Right, Bottom
    var2 = img.copy()
    draw2 = ImageDraw.Draw(var2)
    x1, y1, x2, y2 = raw_box # Интерпретируем иначе
    draw2.rectangle([x1*w/1000, y1*h/1000, x2*w/1000, y2*h/1000], outline="blue", width=5)

    # Вариант 3: Если вдруг масштаб 416, а не 1000
    var3 = img.copy()
    draw3 = ImageDraw.Draw(var3)
    y1, x1, y2, x2 = raw_box
    draw3.rectangle([x1, y1, x2, y2], outline="green", width=5)

    canvas.paste(var1, (0, 0))
    canvas.paste(var2, (w, 0))
    canvas.paste(var3, (w*2, 0))
    
    canvas.save(DEBUG_OUTPUT)
    print(f"Debug image saved to {DEBUG_OUTPUT}. Check Red (current), Blue (swapped), Green (absolute).")

if __name__ == "__main__":
    main()
