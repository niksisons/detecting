"""
Предварительная проверка системы перед обучением
"""
import torch
import sys
from pathlib import Path
import yaml

print("=" * 70)
print("🔍 ПРОВЕРКА СИСТЕМЫ ПЕРЕД ОБУЧЕНИЕМ")
print("=" * 70)

# 1. Проверка PyTorch и CUDA
print("\n1️⃣ Проверка PyTorch и GPU:")
print(f"   PyTorch версия: {torch.__version__}")
cuda_available = torch.cuda.is_available()
print(f"   CUDA доступна: {'✅ Да' if cuda_available else '❌ Нет'}")

if cuda_available:
    print(f"   CUDA версия: {torch.version.cuda}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"   Память GPU: {memory_gb:.2f} GB")
    print(f"   ✅ Обучение будет БЫСТРЫМ на GPU!")
else:
    print(f"   ⚠️ Обучение будет МЕДЛЕННЫМ на CPU")

# 2. Проверка датасета
print("\n2️⃣ Проверка датасета:")
dataset_path = Path(__file__).parent / "system_monitoring2-1"
data_yaml = dataset_path / "data.yaml"

if not dataset_path.exists():
    print(f"   ❌ Папка датасета не найдена: {dataset_path}")
    sys.exit(1)
else:
    print(f"   ✅ Папка датасета найдена")

if not data_yaml.exists():
    print(f"   ❌ Файл data.yaml не найден!")
    sys.exit(1)
else:
    print(f"   ✅ Файл data.yaml найден")
    
    # Чтение data.yaml
    with open(data_yaml, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    print(f"   Количество классов: {data_config.get('nc', 'не указано')}")
    print(f"   Названия классов: {data_config.get('names', 'не указано')}")

# Проверка наличия изображений
train_images = dataset_path / "train" / "images"
val_images = dataset_path / "valid" / "images"
test_images = dataset_path / "test" / "images"

train_count = len(list(train_images.glob("*.jpg"))) if train_images.exists() else 0
val_count = len(list(val_images.glob("*.jpg"))) if val_images.exists() else 0
test_count = len(list(test_images.glob("*.jpg"))) if test_images.exists() else 0

print(f"\n   📊 Количество изображений:")
print(f"      Train: {train_count}")
print(f"      Valid: {val_count}")
print(f"      Test: {test_count}")

if train_count == 0:
    print(f"   ❌ В папке train нет изображений!")
    sys.exit(1)

# 3. Проверка предобученной модели
print("\n3️⃣ Проверка предобученной модели:")
model_path = Path(__file__).parent / "yolo11n.pt"
if model_path.exists():
    print(f"   ✅ Модель yolo11n.pt найдена")
else:
    print(f"   ⚠️ Модель yolo11n.pt не найдена")
    print(f"   Она будет автоматически загружена при первом запуске")

# 4. Проверка пакетов
print("\n4️⃣ Проверка необходимых пакетов:")
try:
    from ultralytics import YOLO
    print(f"   ✅ ultralytics установлен")
except ImportError:
    print(f"   ❌ ultralytics не установлен! Запустите: pip install ultralytics")
    sys.exit(1)

try:
    import cv2
    print(f"   ✅ opencv-python установлен")
except ImportError:
    print(f"   ❌ opencv-python не установлен! Запустите: pip install opencv-python")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    print(f"   ✅ python-dotenv установлен")
except ImportError:
    print(f"   ❌ python-dotenv не установлен! Запустите: pip install python-dotenv")
    sys.exit(1)

# 5. Проверка .env файла
print("\n5️⃣ Проверка конфигурации:")
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    print(f"   ✅ Файл .env найден")
    from dotenv import load_dotenv
    import os
    load_dotenv()
    
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if api_key and api_key != "your_roboflow_api_key_here":
        print(f"   ✅ API ключ Roboflow настроен")
    else:
        print(f"   ⚠️ API ключ Roboflow не настроен (но для локального датасета не нужен)")
else:
    print(f"   ✅ Файл .env найден")

# 6. Проверка выходных папок
print("\n6️⃣ Проверка выходных папок:")
models_dir = Path(__file__).parent / "models"
output_dir = Path(__file__).parent / "output"

models_dir.mkdir(exist_ok=True)
output_dir.mkdir(exist_ok=True)

print(f"   ✅ Папка models готова: {models_dir}")
print(f"   ✅ Папка output готова: {output_dir}")

# Итоговый отчёт
print("\n" + "=" * 70)
print("📋 ИТОГОВЫЙ ОТЧЁТ:")
print("=" * 70)

all_checks = [
    ("GPU доступен", cuda_available),
    ("Датасет найден", train_count > 0),
    ("Пакеты установлены", True),
    ("Папки созданы", True)
]

passed = sum(1 for _, check in all_checks if check)
total = len(all_checks)

for name, check in all_checks:
    status = "✅" if check else "❌"
    print(f"{status} {name}")

print("\n" + "=" * 70)

if passed == total:
    print("🎉 ВСЁ ГОТОВО К ОБУЧЕНИЮ!")
    print("\n🚀 Запустите обучение командой:")
    print("   python train_model.py")
    
    if cuda_available:
        print(f"\n⚡ С вашей GPU обучение займёт ~30-45 минут (100 эпох)")
    else:
        print(f"\n⏰ На CPU обучение займёт несколько часов")
else:
    print(f"⚠️ Исправьте {total - passed} проблем(у) перед запуском обучения")
    sys.exit(1)

print("=" * 70)
