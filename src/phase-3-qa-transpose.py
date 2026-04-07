import json
import random
import re
from pathlib import Path
from PIL import Image, ImageDraw

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
VQA_LABEL_FILE = PROCESSED_DIR / "vqa_labels.json" # Используем локальный результат теста
OUTPUT_QA_IMAGE = PROCESSED_DIR / "qa_grid_transposed.jpg"

def parse_boxes(text):
    try:
        clean_text = re.sub(r'```json|```', '', text).strip()
        data = json.loads(clean_text)
        return [item['bbox_2d'] for item in data if 'bbox_2d' in item]
    except:
        found = re.findall(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', text)
        return [[int(x) for x in b] for b in found]

def main():
    if not VQA_LABEL_FILE.exists():
        print(f"Error: {VQA_LABEL_FILE} not found!")
        return

    with open(VQA_LABEL_FILE, "r") as f:
        vqa_results = json.load(f)

    selected_images = list(vqa_results.keys())[:15]
    
    cols, rows = 5, 3
    thumb_size = (300, 300)
    canvas = Image.new("RGB", (cols * thumb_size[0], rows * thumb_size[1]), "white")

    for i, img_name in enumerate(selected_images):
        src_path = next(RAW_DATA_DIR.rglob(img_name), None)
        if not src_path: continue
            
        img = Image.open(src_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        boxes = parse_boxes(vqa_results[img_name])
        
        for box in boxes:
            # ПРИМЕНЯЕМ ГИПОТЕЗУ ТРАНСПОНИРОВАНИЯ (X <-> Y)
            # Если оригинал: val1, val2, val3, val4
            val1, val2, val3, val4 = box
            
            # Если это Ymin, Xmin, Ymax, Xmax, то меняем на Xmin, Ymin, Xmax, Ymax
            left, top, right, bottom = val2, val1, val4, val3
            
            draw.rectangle([left, top, right, bottom], outline="green", width=5)
        
        img.thumbnail(thumb_size)
        x_pos = (i % cols) * thumb_size[0]
        y_pos = (i // cols) * thumb_size[1]
        canvas.paste(img, (x_pos, y_pos))
        
    canvas.save(OUTPUT_QA_IMAGE)
    print(f"Transposed QA Grid saved to {OUTPUT_QA_IMAGE}")

if __name__ == "__main__":
    main()
