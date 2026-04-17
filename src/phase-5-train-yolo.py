import os
from ultralytics import YOLO
import mlflow
from ultralytics import settings

# Отключаем интеграции, которые не нужны
settings.update({'clearml': False, 'comet': False, 'wandb': False})
os.environ['REPORT_TO'] = 'mlflow'

# НАСТРОЙКИ MLFLOW
MLFLOW_TRACKING_URI = "http://188.243.201.66:5000/"  # Указано в задании
PROJECT_NAME = "HelmetDetection_Synthetic"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(PROJECT_NAME)

# ПУТИ
DATA_YAML = "data/yolo_dataset/data.yaml"

def main():
    # 1. Проверяем наличие данных
    if not os.path.exists(DATA_YAML):
        print(f"Error: {DATA_YAML} not found! Run Phase 4 first.")
        return

    # 2. Инициализируем YOLOv8m (Medium)
    # На 1080 Ti (11Gb) она отлично поместится даже на 640px
    model = YOLO("yolov8m.pt") 

    print(f"Starting training on device: {model.device}")
    
    # 3. Обучение с автоматическим логированием в MLFlow
    results = model.train(
        data=DATA_YAML,
        epochs=50,
        imgsz=640,
        batch=16,
        project=PROJECT_NAME,
        name="v8m_vqa_labels_640",
        device=0, # Наша 1080 Ti
        patience=10, # Early stopping
        save=True,
        cache=True # Ускоряет чтение с диска
    )

    print("Training complete!")
    print(f"Best metrics: {results.results_dict}")

if __name__ == "__main__":
    main()
