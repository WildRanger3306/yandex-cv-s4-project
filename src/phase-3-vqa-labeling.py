import os
os.environ["VLLM_USE_V1"] = "0"
import json
from pathlib import Path

# Необходимо убедиться, что qwen_vl_utils установлен в venv
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("Please install qwen_vl_utils (e.g., pip install qwen-vl-utils)")
    exit(1)

from transformers import AutoProcessor
from vllm import LLM, SamplingParams

DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = PROCESSED_DIR / "vqa_labels.json"

# Конфигурация модели на базе файлов лекций (lesson-4/task-1.py)
path = "Qwen/Qwen2.5-VL-3B-Instruct"
max_num_seq = 2
max_image_tokens = 1024 
max_text_tokens = 256
max_model_len = max_image_tokens + max_text_tokens + 512

max_gpu_mem_allowed = 11
total_gpu_mem = 80
gpu_memory_utilization = max_gpu_mem_allowed / total_gpu_mem

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

def generate_vlm_messages(image_path: str, system_prompt: str, user_prompt: str):
    messages = [{
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
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
    image_paths = image_paths[:5] # ВРЕМЕННО: берем только 5 картинок для теста
    print(f"Total images for inference: {len(image_paths)}")
    
    if not image_paths:
        return

    print(f"Loading Processor from {path}...")
    processor = AutoProcessor.from_pretrained(path, model_max_length=max_model_len)
    
    print("Loading vLLM module. This may take a moment...")
    llm = LLM(
        model=path,
        max_num_seqs=max_num_seq,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization
    )
    
    sampling_params = SamplingParams(temperature=0.1, max_tokens=256)
    
    # Qwen-VL отлично детектит, когда ее просят найти объекты
    system_prompt = "You are a professional safety inspector. Provide accurate bounding boxes."
    user_prompt = "Find all people NOT wearing a safety helmet. Return bounding boxes for them."
    
    print("Preparing inputs...")
    inputs_for_vllm = []
    
    for img_path in image_paths:
        messages = generate_vlm_messages(str(img_path.absolute()), system_prompt, user_prompt)
        inputs = prepare_inputs_for_vllm(messages, processor)
        inputs_for_vllm.append(inputs)
        
    print(f"Starting batch VQA generation on {len(inputs_for_vllm)} images...")
    # vLLM will automatically batch internally and show a progress bar
    outputs = llm.generate(inputs_for_vllm, sampling_params=sampling_params)
    
    results = {}
    for img_path, output in zip(image_paths, outputs):
        generated_text = output.outputs[0].text
        results[img_path.name] = generated_text
        
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print(f"Inference complete. Raw labels saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
