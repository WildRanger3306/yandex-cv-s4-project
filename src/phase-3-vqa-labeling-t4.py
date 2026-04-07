import os
import json
from pathlib import Path

# В Colab обычно нужно:
# !pip install vllm qwen-vl-utils

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("Please install qwen_vl_utils: pip install qwen-vl-utils")
    exit(1)

from transformers import AutoProcessor
from vllm import LLM, SamplingParams

# В Colab пути могут отличаться, адаптируйте под свою структуру
DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = PROCESSED_DIR / "vqa_labels_t4.json"

# Конфигурация модели
# Qwen2.5-VL-3B отлично помещается в 16Гб T4
path = "Qwen/Qwen2.5-VL-3B-Instruct"
max_num_seq = 4 # T4 позволяет больше параллелизма
max_image_tokens = 1024 
max_text_tokens = 256
max_model_len = 2048 # Увеличиваем контекст

# Для T4 (16GB)
gpu_memory_utilization = 0.90 

def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )
    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs
    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }

def generate_vlm_messages(image_path: str, user_prompt: str):
    messages = [{
            "role": "system",
            "content": [{"type": "text", "text": "You are a professional safety inspector. Provide accurate bounding boxes."}]
        }, {
            "role": "user",
            "content": [
                {"type": "image", "max_pixels": max_image_tokens * 28 * 28, "image": f"file://{image_path}"}, 
                {"type": "text", "text": user_prompt}
            ]
        }]
    return messages 

def main():
    image_paths = list(DATA_DIR.rglob("*.jpg")) + list(DATA_DIR.rglob("*.png"))
    print(f"Total images for T4 inference: {len(image_paths)}")
    
    if not image_paths:
        print("No images found! Check DATA_DIR.")
        return

    print(f"Loading Processor and vLLM Engine (T4 Mode)...")
    processor = AutoProcessor.from_pretrained(path, model_max_length=max_model_len)
    
    # На T4 vLLM должен работать "из коробки"
    llm = LLM(
        model=path,
        max_num_seqs=max_num_seq,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True
    )
    
    sampling_params = SamplingParams(temperature=0.1, max_tokens=256)
    user_prompt = "Find all people NOT wearing a safety helmet. Return bounding boxes [ymin, xmin, ymax, xmax] for them."
    
    print("Preparing inputs...")
    inputs_for_vllm = []
    for img_path in image_paths:
        messages = generate_vlm_messages(str(img_path.absolute()), user_prompt)
        inputs = prepare_inputs_for_vllm(messages, processor)
        inputs_for_vllm.append(inputs)
        
    print(f"Starting batch VQA on T4 for {len(inputs_for_vllm)} images...")
    outputs = llm.generate(inputs_for_vllm, sampling_params=sampling_params)
    
    results = {}
    for img_path, output in zip(image_paths, outputs):
        results[img_path.name] = output.outputs[0].text
        
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print(f"Done! T4 inference results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
