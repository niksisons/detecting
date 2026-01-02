# 📦 Зависимости проекта Discipline Monitor

## Установка

```bash
# Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate  # Linux/Mac

# Установить зависимости
pip install -r requirements.txt
```

## Файлы зависимостей

| Файл | Назначение |
|------|-----------|
| `requirements.txt` | Полная установка |
| `requirements-minimal.txt` | Минимальная установка |
| `requirements-gpu.txt` | Для NVIDIA GPU |
| `requirements-dev.txt` | Для разработки |
| `requirements_cloud.txt` | Для Streamlit Cloud |

## Основные библиотеки

- **ultralytics** - YOLO детекция нарушений
- **face-recognition** - Распознавание лиц (dlib)
- **opencv-python** - Обработка видео
- **streamlit** - Веб-интерфейс
- **torch/torchvision** - PyTorch для нейросетей

## Установка dlib (если проблемы)

```bash
# Linux
sudo apt-get install cmake
pip install dlib face-recognition

# Windows
pip install cmake
pip install dlib face-recognition
```
