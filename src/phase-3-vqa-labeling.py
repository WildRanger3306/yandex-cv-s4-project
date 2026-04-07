import os
import json
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = PROCESSED_DIR / "vqa_labels.json"

# Конфигурация модели
# Qwen2.5-VL-3B занимает ~6GB в float16, что отлично подходит для 11GB VRAM
model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

def main():
    image_paths = list(DATA_DIR.rglob("*.jpg")) + list(DATA_DIR.rglob("*.png"))
    image_paths = image_paths[:5] # ТЕСТ: берем только 5 картинок для проверки
    print(f"Total images for tagging (Transformers): {len(image_paths)}")
    
    if not image_paths:
        return

    print(f"Loading Model and Processor: {model_id}...")
    # Используем float16 для Pascal (1080Ti не поддерживает bfloat16)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    # Отключаем fast-процессор для совместимости с torch 2.2.0
    processor = AutoProcessor.from_pretrained(model_id, use_fast=False)

    # Настройка промпта
    user_prompt = "Find all people NOT wearing a safety helmet. Return bounding boxes in format [ymin, xmin, ymax, xmax] for each."
    
    results = {}
    
    print("Starting inference...")
    for img_path in tqdm(image_paths):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{img_path.absolute()}"},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]

        # Подготовка входов
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        # Генерация
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=256)
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        results[img_path.name] = output_text[0]
        
    # Сохранение результатов
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print(f"Inference complete. Labels saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
