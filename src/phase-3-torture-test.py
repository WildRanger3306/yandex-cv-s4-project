import json
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

RAW_DATA_DIR = Path("data/raw/images")
PROCESSED_DIR = Path("data/processed")
VQA_LABEL_FILE = PROCESSED_DIR / "vqa_labels.json"
TORTURE_OUTPUT = PROCESSED_DIR / "torture_test.jpg"

def parse_boxes(text):
    try:
        clean_text = re.sub(r'```json|```', '', text).strip()
        data = json.loads(clean_text)
        return [item['bbox_2d'] for item in data if 'bbox_2d' in item]
    except:
        found = re.findall(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', text)
        return [[int(x) for x in b] for b in found]

def main():
    img_name = "hard_hat_workers10.png"
    src_path = RAW_DATA_DIR / img_name
    
    with open(VQA_LABEL_FILE, "r") as f:
        vqa_results = json.load(f)
    
    raw_boxes = parse_boxes(vqa_results[img_name])
    img = Image.open(src_path).convert("RGB")
    w, h = img.size
    
    # Сетка 3x2
    cols, rows = 3, 2
    canvas = Image.new("RGB", (w * cols, h * rows), "white")
    
    variants = [
        ("1. Original YXYX", lambda b: (b[1], b[0], b[3], b[2])),
        ("2. Transposed XYXY", lambda b: (b[0], b[1], b[2], b[3])),
        ("3. V-Flip", lambda b: (b[1], h - b[2], b[3], h - b[0])),
        ("4. H-Flip", lambda b: (w - b[3], b[0], w - b[1], b[2])),
        ("5. Rot90 CCW", lambda b: (b[0], w - b[3], b[2], w - b[1])),
        ("6. Diagonal Flip", lambda b: (h - b[2], w - b[3], h - b[0], w - b[1]))
    ]
    
    for i, (label, transform) in enumerate(variants):
        temp_img = img.copy()
        draw = ImageDraw.Draw(temp_img)
        
        for box in raw_boxes:
            # Преобразуем бокc согласно гипотезе
            x1, y1, x2, y2 = transform(box)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            
        # Добавляем подпись
        draw.text((10, 10), label, fill="yellow")
        
        x_pos = (i % cols) * w
        y_pos = (i // cols) * h
        canvas.paste(temp_img, (x_pos, y_pos))
        
    canvas.save(TORTURE_OUTPUT)
    print(f"Torture test saved to {TORTURE_OUTPUT}")

if __name__ == "__main__":
    main()
