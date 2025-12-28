"""
Скрипт для дообучения модели YOLO11 на датасете с Roboflow
"""
import os
from pathlib import Path
from ultralytics import YOLO
from roboflow import Roboflow
from dotenv import load_dotenv
import config

# Загрузка переменных окружения
load_dotenv()


def download_dataset():
    """Загрузка датасета с Roboflow"""
    print("📥 Проверка датасета...")
    
    # Путь к папке датасета
    dataset_name = f"{config.ROBOFLOW_PROJECT}-{config.ROBOFLOW_VERSION}"
    dataset_path = config.DATA_DIR / dataset_name
    data_yaml = dataset_path / "data.yaml"
    
    # Проверяем, существует ли датасет
    if dataset_path.exists() and data_yaml.exists():
        print(f"✅ Датасет уже загружен: {dataset_path}")
        print(f"✅ Найден data.yaml: {data_yaml}")
        return str(dataset_path)
    
    # Если датасета нет - скачиваем
    print(f"📥 Датасет не найден. Начинаем загрузку с Roboflow...")
    
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key or api_key == "your_roboflow_api_key_here":
        raise ValueError("❌ Установите ROBOFLOW_API_KEY в файле .env")
    
    try:
        rf = Roboflow(api_key=api_key)
        workspace = config.ROBOFLOW_WORKSPACE
        project_name = config.ROBOFLOW_PROJECT
        version = config.ROBOFLOW_VERSION
        
        print(f"🔗 Подключение к Roboflow:")
        print(f"   Workspace: {workspace}")
        print(f"   Project: {project_name}")
        print(f"   Version: {version}")
        
        project = rf.workspace(workspace).project(project_name)
        dataset = project.version(version).download("yolov11", location=str(config.DATA_DIR))
        
        print(f"✅ Датасет загружен в: {dataset.location}")
        
        # Ищем data.yaml в загруженной директории
        downloaded_path = Path(dataset.location)
        downloaded_yaml = downloaded_path / "data.yaml"
        
        if downloaded_yaml.exists():
            print(f"✅ Найден data.yaml: {downloaded_yaml}")
            return str(downloaded_path)
        
        # Иногда датасет загружается во вложенную папку
        for subdir in downloaded_path.iterdir():
            if subdir.is_dir():
                potential_yaml = subdir / "data.yaml"
                if potential_yaml.exists():
                    print(f"✅ Найден data.yaml в: {potential_yaml}")
                    return str(subdir)
        
        raise FileNotFoundError(f"❌ Не удалось найти data.yaml в {downloaded_path}")
        
    except Exception as e:
        print(f"❌ Ошибка при загрузке датасета: {e}")
        print(f"\n💡 Проверьте:")
        print(f"   1. API ключ в .env файле")
        print(f"   2. Название workspace: {config.ROBOFLOW_WORKSPACE}")
        print(f"   3. Название проекта: {config.ROBOFLOW_PROJECT}")
        print(f"   4. Версию датасета: {config.ROBOFLOW_VERSION}")
        raise


def train_yolo_model(dataset_path=None):
    """Обучение модели YOLO11"""
    print("\n🚀 Начало обучения модели YOLO11...")
    
    # Проверка доступности GPU
    import torch
    device = "cuda:0" if torch.cuda.is_available() and config.USE_GPU else "cpu"
    
    if device == "cuda:0":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎮 GPU обнаружена: {gpu_name}")
        print(f"💾 Память GPU: {gpu_memory:.2f} GB")
        print(f"✅ Обучение будет на GPU")
    else:
        print(f"⚠️ ВНИМАНИЕ: GPU не обнаружена или отключена")
        print(f"📊 Используем CPU для обучения (будет медленнее)")
    
    # Загрузка предобученной модели
    model = YOLO(config.YOLO_MODEL)
    
    # Путь к data.yaml
    if dataset_path is None:
        # Пробуем найти локальный датасет или скачать с Roboflow
        local_dataset = config.DATA_DIR / f"{config.ROBOFLOW_PROJECT}-{config.ROBOFLOW_VERSION}"
        if local_dataset.exists() and (local_dataset / "data.yaml").exists():
            dataset_path = local_dataset
            print(f"📂 Используется локальный датасет: {dataset_path}")
        else:
            # Если локального нет - скачиваем
            dataset_path = download_dataset()
    
    data_yaml = Path(dataset_path) / "data.yaml"
    
    if not data_yaml.exists():
        raise FileNotFoundError(f"❌ Файл {data_yaml} не найден!")
    
    print(f"📄 Используется data.yaml: {data_yaml}")
    
    # Параметры обучения (оптимизированы для GPU/CPU)
    if device == "cuda:0":
        epochs = config.TRAIN_EPOCHS
        batch = config.TRAIN_BATCH
        imgsz = config.TRAIN_IMG_SIZE
        print(f"🎯 Параметры для GPU: эпохи={epochs}, batch={batch}, размер={imgsz}")
    else:
        epochs = 20
        batch = 8  # Для CPU
        imgsz = 640
        print(f"🎯 Параметры для CPU: эпохи={epochs}, batch={batch}, размер={imgsz}")
    
    # Обучение модели
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=config.TRAIN_PATIENCE,
        save=True,
        project=str(config.MODELS_DIR),
        name="discipline_monitor",
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        verbose=True,
        seed=42,
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=False,
        close_mosaic=10,
        resume=False,
        amp=True,
        fraction=1.0,
        profile=False,
        freeze=None,
        device=device
    )
    
    print("\n✅ Обучение завершено!")
    print(f"📊 Результаты сохранены в: {config.MODELS_DIR / 'discipline_monitor'}")
    
    return results


def validate_model(dataset_path):
    """Валидация обученной модели"""
    print("\n📊 Валидация модели...")
    
    # Путь к лучшей модели
    best_model_path = config.MODELS_DIR / "discipline_monitor" / "weights" / "best.pt"
    
    if not best_model_path.exists():
        print("❌ Модель не найдена! Сначала обучите модель.")
        return
    
    model = YOLO(str(best_model_path))
    
    # Определение устройства
    import torch
    device = "cuda:0" if torch.cuda.is_available() and config.USE_GPU else "cpu"
    print(f"🔍 Валидация на устройстве: {device}")
    
    # Валидация
    data_yaml = Path(dataset_path) / "data.yaml"
    metrics = model.val(data=str(data_yaml), device=device)
    
    print("\n📈 Метрики модели:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")
    
    return metrics


def export_model():
    """Экспорт модели в различные форматы"""
    print("\n📦 Экспорт модели...")
    
    best_model_path = config.MODELS_DIR / "discipline_monitor" / "weights" / "best.pt"
    
    if not best_model_path.exists():
        print("❌ Модель не найдена!")
        return
    
    model = YOLO(str(best_model_path))
    
    # Копирование лучшей модели в корень models
    import shutil
    shutil.copy(best_model_path, config.YOLO_TRAINED_MODEL)
    print(f"✅ Модель скопирована в: {config.YOLO_TRAINED_MODEL}")
    
    # Экспорт в ONNX (опционально, для быстрого инференса)
    # try:
    #     model.export(format="onnx")
    #     print("✅ Модель экспортирована в ONNX формат")
    # except Exception as e:
    #     print(f"⚠️ Не удалось экспортировать в ONNX: {e}")


def main():
    """Главная функция"""
    print("=" * 60)
    print("🎯 ОБУЧЕНИЕ YOLO11 ДЛЯ МОНИТОРИНГА ДИСЦИПЛИНЫ")
    print("=" * 60)
    
    try:
        # Проверка/загрузка датасета
        dataset_path = download_dataset()
        
        # 1. Обучение модели
        train_yolo_model(str(dataset_path))
        
        # 2. Валидация модели
        validate_model(str(dataset_path))
        
        # 3. Экспорт модели
        export_model()
        
        print("\n" + "=" * 60)
        print("🎉 ВСЕ ЭТАПЫ ВЫПОЛНЕНЫ УСПЕШНО!")
        print("=" * 60)
        print(f"\n📍 Обученная модель: {config.YOLO_TRAINED_MODEL}")
        print("\n💡 Следующий шаг: запустите detect_violations.py для детекции")
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
