import os
import json
import torch
import random
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from pathlib import Path

# ПУТИ
DATA_DIR = Path("data/raw/images")
OUTPUT_FILE = Path("data/processed/vqa_labels_pro.json")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# ЛИМИТ (используем ваши 15 картинок)
LIMIT = 15

def main():
    if not DATA_DIR.exists():
        print(f"Error: {DATA_DIR} not found!")
        return

    # Берем те же 15 случайных файлов для чистоты эксперимента
    all_image_files = sorted(list(DATA_DIR.glob("*.png")) + list(DATA_DIR.glob("*.jpg")))
    random.seed(42) # Фиксируем сид, чтобы сравнивать на тех же картинках
    image_files = random.sample(all_image_files, min(LIMIT, len(all_image_files)))
    print(f"Total images for COT debugging (Local 1080Ti): {len(image_files)}")

    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

    print(f"Loading Model: {model_id}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id, use_fast=False)

    # CHAIN-OF-THOUGHT PROMPT
    prompt_text = (
        "Analyze this industrial safety image step-by-step:\n"
        "1. Identify all people/workers visible in the image.\n"
        "2. For each person, look closely at their head and state if they are wearing a hard hat/helmet, a regular cap, or nothing.\n"
        "3. Based on your analysis, provide bounding boxes in [xmin, ymin, xmax, ymax] format (absolute pixels) "
        "ONLY for heads of workers NOT wearing a safety helmet. "
        "Wrap the final boxes in a JSON-like list at the very end of your response."
    )

    results = {}
    print("Starting Pro Inference (COT)...")
    
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

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        
        # Важно: фиксируем разрешение, чтобы мелкие детали не расплывались
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            # Увеличиваем max_new_tokens, чтобы хватило места на рассуждения
            generated_ids = model.generate(**inputs, max_new_tokens=1024)
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        
        # Сохраняем ПОЛНЫЙ текст рассуждений для анализа
        results[img_path.name] = output_text[0]
        
        # Выведем рассуждения для первой картинки в консоль для примера
        if img_path == image_files[0]:
            print(f"\nExample COT for {img_path.name}:\n{output_text[0]}\n")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Pro Inference complete. Full logs saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
