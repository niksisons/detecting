# 🚀 ШПАРГАЛКА: Команды для запуска

## ⚡ БЫСТРЫЙ СТАРТ

### Активация виртуального окружения:
```powershell
.\.venv\Scripts\activate
```

### Проверка системы (СНАЧАЛА ЭТО!):
```powershell
.\.venv\Scripts\python.exe check_system.py
```

### Запуск обучения:
```powershell
.\.venv\Scripts\python.exe train_model.py
```

---

## 📊 Проверки перед обучением

### Проверить GPU:
```powershell
.\.venv\Scripts\python.exe check_gpu.py
```

**Ожидаемый результат:**
```
🎮 GPU обнаружена: NVIDIA GeForce RTX 5070
💾 Память GPU: 11.94 GB
✅ GPU работает корректно!
```

### Проверить датасет:
```powershell
dir system_monitoring2-1\train\images | measure
dir system_monitoring2-1\valid\images | measure
dir system_monitoring2-1\test\images | measure
```

---

## 🎓 Обучение модели

### Полный цикл (проверка + обучение):
```powershell
.\.venv\Scripts\python.exe quick_start.py
```

### Только обучение:
```powershell
.\.venv\Scripts\python.exe train_model.py
```

**Параметры для вашей GPU (автоматически):**
- Эпохи: 100
- Batch size: 32
- Размер: 640x640
- Время: ~30-45 минут

---

## 🎬 Детекция нарушений

### С веб-камеры:
```powershell
.\.venv\Scripts\python.exe detect_violations.py --source 0
```

### Из видеофайла:
```powershell
.\.venv\Scripts\python.exe detect_violations.py --source "путь\к\видео.mp4"
```

### С указанием модели:
```powershell
.\.venv\Scripts\python.exe detect_violations.py --source 0 --model models\best.pt
```

---

## 👤 База данных лиц

### Запуск интерактивного режима:
```powershell
.\.venv\Scripts\python.exe face_database.py
```

**Команды в интерактивном режиме:**
- `add` - добавить студента (одно фото)
- `add_folder` - добавить студента (папка с фото)
- `list` - показать всех студентов
- `remove` - удалить студента
- `exit` - выход

---

## 🔍 Мониторинг GPU

### Постоянный мониторинг (в отдельном терминале):
```powershell
nvidia-smi -l 1
```

### Одноразовая проверка:
```powershell
nvidia-smi
```

**Нормальные показатели при обучении:**
- GPU Utilization: 95-100%
- Memory: 8-10 GB / 12 GB
- Temp: 60-75°C
- Power: 150-200W

---

## 📁 Структура файлов

### Важные файлы:
```
yolo11n.pt                    - Предобученная модель
system_monitoring2-1/         - Датасет
├── data.yaml                 - Конфигурация датасет
├── train/images/             - Обучающие изображения (3624)
├── valid/images/             - Валидационные (216)
└── test/images/              - Тестовые (89)

models/
├── best.pt                   - 🎯 ОБУЧЕННАЯ МОДЕЛЬ (используйте её!)
└── discipline_monitor/
    ├── weights/
    │   ├── best.pt
    │   └── last.pt
    ├── results.png           - Графики обучения
    └── confusion_matrix.png  - Матрица ошибок

output/
├── videos/                   - Записи нарушений
├── faces/                    - Фото нарушителей
└── report_*.json            - Отчеты
```

---

## 🎯 Тестирование модели

### Быстрый тест на одном изображении:
```powershell
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; model = YOLO('models/best.pt'); model.predict('system_monitoring2-1/test/images/100_test_jpg.rf.fd45cec812bf506723ad91d2351f8176.jpg', save=True, conf=0.5)"
```

Результат в `runs/detect/predict/`

### Валидация на всём test датасете:
```powershell
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; model = YOLO('models/best.pt'); model.val(data='system_monitoring2-1/data.yaml', split='test')"
```

---

## ⚙️ Изменение параметров

### В config.py:

```python
# Для быстрого обучения (меньше качество):
TRAIN_EPOCHS = 50
TRAIN_BATCH = 16
TRAIN_IMG_SIZE = 512

# Для максимального качества (дольше):
TRAIN_EPOCHS = 200
TRAIN_BATCH = 16
TRAIN_IMG_SIZE = 1024

# Текущие (баланс):
TRAIN_EPOCHS = 100
TRAIN_BATCH = 32
TRAIN_IMG_SIZE = 640
```

---

## 🆘 Решение проблем

### "CUDA out of memory":
Уменьшите batch в `train_model.py` или `config.py`:
```python
TRAIN_BATCH = 16  # вместо 32
# или
TRAIN_BATCH = 8
```

### "GPU не используется":
```powershell
# Проверка
.\.venv\Scripts\python.exe check_gpu.py

# Если нужно, переустановите PyTorch:
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### "No module named ...":
```powershell
# Убедитесь, что используете venv:
.\.venv\Scripts\activate

# Установите недостающий пакет:
pip install ultralytics opencv-python python-dotenv
```

---

## 📊 Понимание результатов

### После обучения:

**Хорошие метрики:**
- mAP50: > 0.80
- mAP50-95: > 0.60
- Precision: > 0.75
- Recall: > 0.70

**Отличные метрики:**
- mAP50: > 0.90
- mAP50-95: > 0.75
- Precision: > 0.85
- Recall: > 0.80

### Графики в results.png:
- **train/box_loss** - должен падать
- **val/box_loss** - должен падать
- **metrics/mAP50** - должен расти
- **metrics/precision** - должен расти
- **metrics/recall** - должен расти

---

## 🎉 ГОТОВО К ЗАПУСКУ!

### Команда для старта ПРЯМО СЕЙЧАС:

```powershell
.\.venv\Scripts\python.exe train_model.py
```

**Время обучения:** ~30-45 минут на RTX 5070 ⚡

---

## 📞 Полезные ссылки

- **README.md** - Основная документация
- **TRAINING_GUIDE.md** - Подробная инструкция
- **START_HERE.md** - С чего начать
- **TODO.md** - Чеклист задач

---

## ✅ Ваша система ГОТОВА:

- ✅ GPU: NVIDIA GeForce RTX 5070 (11.94 GB)
- ✅ CUDA: 13.0
- ✅ PyTorch: 2.9.1+cu130
- ✅ Датасет: 3624 train + 216 valid + 89 test
- ✅ Классы: food_and_water, sleep_and_phone, something
- ✅ Все пакеты установлены

**НАЧИНАЙТЕ ОБУЧЕНИЕ!** 🚀
