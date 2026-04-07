import os
import random
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def main():
    image_paths = list(DATA_DIR.rglob("*.jpg")) + list(DATA_DIR.rglob("*.png"))
    
    if not image_paths:
        print("No images found in data/raw.")
        return
        
    print(f"Total images found: {len(image_paths)}")
    
    sample_size = min(100, len(image_paths))
    samples_for_stats = random.sample(image_paths, sample_size)
    
    widths = []
    heights = []
    for path in samples_for_stats:
        with Image.open(path) as img:
            widths.append(img.width)
            heights.append(img.height)
            
    print(f"\n--- Statistics (based on {sample_size} samples) ---")
    print(f"Resolutions - Widths: min={min(widths)}, max={max(widths)}, mean={sum(widths)/len(widths):.1f}")
    print(f"Resolutions - Heights: min={min(heights)}, max={max(heights)}, mean={sum(heights)/len(heights):.1f}")
    
    grid_size = 3
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    axes = axes.flatten()
    
    samples_for_plot = random.sample(image_paths, min(9, len(image_paths)))
    
    for ax, path in zip(axes, samples_for_plot):
        img = Image.open(path)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(path.name, fontsize=8)
        
    for j in range(len(samples_for_plot), len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    
    plot_path = PROCESSED_DIR / "eda_samples.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved EDA sample plot to {plot_path}")
    
    print("\n--- Выводы о потенциальных сложностях для VQA (автоматическая генерация разметки) ---")
    print("1. Перекрытия (Occlusions): Люди могут стоять плотно, закрывая друг друга и свои шлемы.")
    print("2. Масштаб (Scale variation): Камеры установлены далеко, лица и шлемы мелкие и шумные.")
    print("3. Сложные ракурсы (Viewpoints): Ракурс съемки сверху означает, что лица часто не видно целиком.")
    print("4. Освещение (Lighting): Тень или яркий прожектор могут изменять цвет строительной каски.")
    print("5. False Positives (Ошибки модели): Головные уборы, кепки, капюшоны или даже строительное оборудование могут быть ошибочно приняты за шлем или отсутствие шлема.")
    print("----------------------------------------------------------------------------------")

if __name__ == "__main__":
    main()
