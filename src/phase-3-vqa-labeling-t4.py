import os
import json
import torch
from vllm import LLM, SamplingParams
from pathlib import Path
from PIL import Image

# ПУТИ (Настройте под Colab)
DATA_DIR = Path("data/raw/images")
OUTPUT_FILE = "vqa_labels_improved.json"

# ЛИМИТ - для тестового прогона берем 100 фото
LIMIT = 1000

# МОДЕЛЬ - Переходим на 7B для повышения качества детекции
# Она лучше видит мелкие детали и реже ошибается в классификации "каска/кепка"
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

def main():
    if not DATA_DIR.exists():
        print(f"Error: {DATA_DIR} not found! Upload your images first.")
        return

    image_files = list(DATA_DIR.glob("*.png")) + list(DATA_DIR.glob("*.jpg"))
    image_files = sorted(image_files)[:LIMIT]
    print(f"Total images to process (Test run): {len(image_files)}")

    # Инициализация vLLM (Для T4/L4 в Colab)
    # Используем float16 или bfloat16 (если L4)
    llm = LLM(
        model=MODEL_ID,
        max_model_len=2048,
        max_num_seqs=4,
        # На L4 можно включить bfloat16, на T4 лучше float16
        dtype="float16", 
        limit_mm_per_prompt={"image": 1}
    )

    sampling_params = SamplingParams(
        temperature=0.01, # Почти детерминированный ответ
        max_tokens=512,
        stop=["<|im_end|>"]
    )

    # Уточненный CoT промпт для 7B модели
    prompt = ("<|im_start|>system\nYou are a professional safety inspector. Analyze the image step-by-step to identify safety violations. "
              "Focus on worker's heads.<|im_end|>\n"
              "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
              "1. Identify all people and their heads.\n"
              "2. For each head, determine if they are wearing a safety helmet, a cap, or nothing.\n"
              "3. List [xmin, ymin, xmax, ymax] boxes ONLY for heads WITHOUT a safety helmet.\n"
              "Provide your reasoning first, then the JSON list of boxes.<|im_end|>\n"
              "<|im_start|>assistant\n")

    # Формируем батч запросов
    prompts = []
    for img_path in image_files:
        prompts.append({
            "prompt": prompt,
            "multi_modal_data": {"image": Image.open(img_path).convert("RGB")}
        })

    print("Starting Batch Inference via vLLM (7B Model)...")
    # Для 7B модели на T4 может потребоваться GPU memory utilization ~0.9
    outputs = llm.generate(prompts, sampling_params)

    # Собираем результаты
    results = {}
    for i, output in enumerate(outputs):
        img_name = image_files[i].name
        results[img_name] = output.outputs[0].text

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Labeling complete! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
