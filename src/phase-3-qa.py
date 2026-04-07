import json
import random
import re
from pathlib import Path
from PIL import Image, ImageDraw

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
VQA_LABEL_FILE = PROCESSED_DIR / "vqa_labels.json"
OUTPUT_QA_IMAGE = PROCESSED_DIR / "qa_grid.jpg"

def parse_boxes(text):
    """Извлекает bbox_2d из JSON-ответа Qwen."""
    try:
        clean_text = re.sub(r'```json|```', '', text).strip()
        data = json.loads(clean_text)
        return [item['bbox_2d'] for item in data if 'bbox_2d' in item]
    except:
        # Fallback на регулярки
        found = re.findall(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', text)
        return [[int(x) for x in b] for b in found]

def main():
    if not VQA_LABEL_FILE.exists():
        print(f"Error: {VQA_LABEL_FILE} not found!")
        return

    with open(VQA_LABEL_FILE, "r") as f:
        vqa_results = json.load(f)

    # Выбираем 15 случайных изображений, у которых есть разметка
    available_images = [img for img in vqa_results.keys() if len(parse_boxes(vqa_results[img])) > 0]
    if len(available_images) < 15:
        available_images = list(vqa_results.keys())
    
    selected_images = random.sample(available_images, min(15, len(available_images)))
    
    # Размеры сетки
    cols, rows = 5, 3
    thumb_size = (300, 300)
    canvas = Image.new("RGB", (cols * thumb_size[0], rows * thumb_size[1]), "white")
    draw_canvas = ImageDraw.Draw(canvas)

    for i, img_name in enumerate(selected_images):
        src_path = next(RAW_DATA_DIR.rglob(img_name), None)
        if not src_path:
            continue
            
        img = Image.open(src_path).convert("RGB")
        w, h = img.size
        
        # Рисуем боксы ПРЯМО на объекте Image (в масштабе оригинала)
        draw = ImageDraw.Draw(img)
        boxes = parse_boxes(vqa_results[img_name])
        
        for box in boxes:
            # ПОДТВЕРЖДЕНО: Модель возвращает [xmin, ymin, xmax, ymax] в ПИКСЕЛЯХ
            xmin, ymin, xmax, ymax = box
            
            # Рисуем зеленый прямоугольник для успеха
            draw.rectangle([xmin, ymin, xmax, ymax], outline="green", width=3)
        
        # Делаем превью для сетки
        img.thumbnail(thumb_size)
        
        # Вычисляем позицию в сетке
        x_pos = (i % cols) * thumb_size[0]
        y_pos = (i // cols) * thumb_size[1]
        
        # Вставляем в канвас
        canvas.paste(img, (x_pos, y_pos))
        
    canvas.save(OUTPUT_QA_IMAGE)
    print(f"QA Grid saved to {OUTPUT_QA_IMAGE}")

if __name__ == "__main__":
    main()
