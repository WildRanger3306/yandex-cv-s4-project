import os
import json
import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from pathlib import Path

# ПУТИ
DATA_DIR = Path("data/raw/images")
OUTPUT_FILE = Path("data/processed/vqa_labels.json")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# ЛИМИТ ДЛЯ ТЕСТА (на 1080 Ti)
LIMIT = 5

def main():
    if not DATA_DIR.exists():
        print(f"Error: {DATA_DIR} not found!")
        return

    # Получаем список файлов
    image_files = sorted(list(DATA_DIR.glob("*.png")) + list(DATA_DIR.glob("*.jpg")))
    image_files = image_files[:LIMIT]
    print(f"Total images for tagging (Local 1080Ti): {len(image_files)}")

    # Идентификатор модели
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

    print(f"Loading Model: {model_id}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16, # Используем float16 для Pascal
        device_map="auto"
    )
    
    # Отключаем fast-процессор для совместимости с torch 2.2.0
    processor = AutoProcessor.from_pretrained(model_id, use_fast=False)

    # Уточненный промпт для детекции именно ГОЛОВ
    prompt_text = "Detect all heads of workers. If a head is NOT wearing a safety helmet, return [ymin, xmin, ymax, xmax] coordinates."

    results = {}
    print("Starting inference...")
    
    for img_path in tqdm(image_files):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(img_path)},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        # Подготовка промпта через шаблоны Qwen
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=256)
            
        # Обрезаем вводные данные из ответа
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        
        results[img_path.name] = output_text[0]

    # Сохраняем результат
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Inference complete. Labels saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
