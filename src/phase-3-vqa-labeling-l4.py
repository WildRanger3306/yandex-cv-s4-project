import os
import json
import torch
from vllm import LLM, SamplingParams
from pathlib import Path
from PIL import Image

# ПУТИ
DATA_DIR = Path("data/raw/images")
OUTPUT_FILE = "vqa_labels_improved_l4.json"

# ЛИМИТЫ
LIMIT = 2000     # L4 мощнее T4, можно брать больше
BATCH_SIZE = 100 # Чанки для защиты системной RAM (не загружаем все фото сразу)

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

    # Инициализация vLLM (Оптимизировано для L4 в Colab/Vertex)
    # На L4 используем bfloat16 для максимальной производительности
    llm = LLM(
        model=MODEL_ID,
        max_model_len=4096,      # Позволяет обрабатывать более сложные сцены
        max_num_seqs=16,         # Увеличиваем параллелизм для 24GB VRAM
        dtype="bfloat16",        # Нативная поддержка BF16 на архитектуре Ada Lovelace
        gpu_memory_utilization=0.9,
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

    results = {}
    
    # Обработка чанками для предотвращения переполнения RAM
    for i in range(0, len(image_files), BATCH_SIZE):
        batch_files = image_files[i : i + BATCH_SIZE]
        print(f"Processing batch {i//BATCH_SIZE + 1} ({len(batch_files)} images)...")
        
        prompts = []
        for img_path in batch_files:
            prompts.append({
                "prompt": prompt,
                "multi_modal_data": {"image": Image.open(img_path).convert("RGB")}
            })

        print(f"Running generation for batch {i//BATCH_SIZE + 1}...")
        outputs = llm.generate(prompts, sampling_params)

        for j, output in enumerate(outputs):
            img_name = batch_files[j].name
            results[img_name] = output.outputs[0].text
        
        # Периодическое сохранение
        with open(OUTPUT_FILE, "w") as f:
            json.dump(results, f, indent=2)

    print(f"Labeling complete! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
