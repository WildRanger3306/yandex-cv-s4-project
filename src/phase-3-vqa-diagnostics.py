import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os

# Пути
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
TEST_IMAGE = "data/raw/images/hard_hat_workers330.png"

def main():
    if not os.path.exists(TEST_IMAGE):
        print(f"ERROR: Image {TEST_IMAGE} not found!")
        return

    print(f"Loading model {MODEL_ID} (float16)...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    # Используем use_fast=False для совместимости с нашим окружением
    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=False)

    print(f"Testing image: {TEST_IMAGE}")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": TEST_IMAGE},
                {
                    "type": "text", 
                    "text": "Find and locate all heads of workers. Return only bounding boxes in format [ymin, xmin, ymax, xmax]."
                }
            ],
        }
    ]

    # Подготовка промпта
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    
    inputs = processor(
        text=[text], 
        images=image_inputs, 
        padding=True, 
        return_tensors="pt"
    ).to(model.device)

    print("Generating response...")
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    
    # Декодируем только ответ
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )

    print("\n--- RAW VLM OUTPUT ---")
    print(output_text[0])
    print("----------------------\n")

if __name__ == "__main__":
    main()
