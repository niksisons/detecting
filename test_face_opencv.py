"""
Тестовый скрипт для проверки распознавания лиц
Позволяет быстро протестировать работу системы
"""
import cv2
import config
from pathlib import Path

# Импортируем базу данных лиц (OpenCV версия)
from face_database_opencv import FaceDatabase


def test_add_face_from_camera():
    """Добавить своё лицо в базу через камеру"""
    print("=" * 60)
    print("ДОБАВЛЕНИЕ ЛИЦА В БАЗУ ДАННЫХ")
    print("=" * 60)
    
    db = FaceDatabase()
    
    name = input("\nВведите ваше имя: ").strip()
    if not name:
        print("Имя не может быть пустым!")
        return
    
    print(f"\nСейчас будет сделано фото для {name}")
    print("Нажмите ПРОБЕЛ чтобы сделать фото, Q для выхода")
    print("-" * 40)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть камеру")
        return
    
    photos_saved = 0
    temp_dir = Path(config.DATA_DIR) / "temp_faces"
    temp_dir.mkdir(exist_ok=True)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Детектируем лица для предпросмотра
        faces = db.detect_faces(frame)
        
        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Инструкция на экране
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "SPACE - save, Q - exit", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Saved: {photos_saved}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("Add Face - SPACE to capture", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Пробел - сохранить фото
            if faces:
                photo_path = temp_dir / f"{name}_{photos_saved}.jpg"
                cv2.imwrite(str(photo_path), frame)
                
                if db.add_person(name, str(photo_path)):
                    photos_saved += 1
                    print(f"  Фото {photos_saved} сохранено!")
                else:
                    print("  Не удалось добавить фото")
                    
                # Удаляем временный файл
                photo_path.unlink(missing_ok=True)
            else:
                print("  Лицо не найдено на кадре!")
        
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if photos_saved > 0:
        db.save_database()
        print(f"\n✅ {name} добавлен в базу ({photos_saved} фото)")
    else:
        print("\n❌ Ни одного фото не добавлено")


def test_recognition_live():
    """Тест распознавания в реальном времени"""
    print("=" * 60)
    print("ТЕСТ РАСПОЗНАВАНИЯ ЛИЦ В РЕАЛЬНОМ ВРЕМЕНИ")
    print("=" * 60)
    
    db = FaceDatabase()
    
    persons = db.list_persons()
    print(f"\nЛюди в базе: {persons if persons else 'база пуста'}")
    print(f"Порог распознавания: {config.FACE_RECOGNITION_TOLERANCE}")
    
    if not persons:
        print("\n⚠️ База пуста! Сначала добавьте людей командой 'add'")
        return
    
    print("\nНажмите Q для выхода")
    print("-" * 40)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть камеру")
        return
    
    frame_count = 0
    last_faces = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Распознаём каждые 3 кадра для производительности
        if frame_count % 3 == 0:
            last_faces = db.recognize_faces_in_frame(frame)
        
        # Рисуем результаты
        for face in last_faces:
            x1, y1, x2, y2 = face["bbox"]
            name = face["name"]
            conf = face["confidence"]
            dist = face["distance"]
            
            # Цвет: зелёный если распознан, красный если нет
            if name != "Unknown":
                color = (0, 255, 0)
                label = f"{name} ({conf:.0%})"
            else:
                color = (0, 0, 255)
                label = f"Unknown (dist: {dist:.2f})"
            
            # Рамка
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Подпись
            cv2.rectangle(frame, (x1, y1 - 30), (x2, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Статус
        cv2.putText(frame, f"Faces: {len(last_faces)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Q - exit", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Face Recognition Test - Q to exit", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nТест завершён")


def test_add_from_file():
    """Добавить лицо из файла"""
    print("=" * 60)
    print("ДОБАВЛЕНИЕ ЛИЦА ИЗ ФАЙЛА")
    print("=" * 60)
    
    db = FaceDatabase()
    
    name = input("\nВведите имя: ").strip()
    if not name:
        print("Имя не может быть пустым!")
        return
    
    image_path = input("Путь к фото: ").strip()
    
    if not Path(image_path).exists():
        print(f"Файл не найден: {image_path}")
        return
    
    if db.add_person(name, image_path):
        db.save_database()
        print(f"\n✅ {name} добавлен в базу")
    else:
        print(f"\n❌ Не удалось добавить {name}")


def show_database():
    """Показать содержимое базы"""
    print("=" * 60)
    print("СОДЕРЖИМОЕ БАЗЫ ДАННЫХ ЛИЦ")
    print("=" * 60)
    
    db = FaceDatabase()
    persons = db.list_persons()
    
    if not persons:
        print("\nБаза пуста")
    else:
        print(f"\nВсего в базе: {len(persons)} человек")
        for i, person in enumerate(persons, 1):
            count = db.get_person_count(person)
            print(f"  {i}. {person} ({count} эмбеддингов)")


def main():
    """Главное меню"""
    print("\n" + "=" * 60)
    print("🎯 ТЕСТ СИСТЕМЫ РАСПОЗНАВАНИЯ ЛИЦ")
    print("=" * 60)
    
    while True:
        print("\nВыберите действие:")
        print("1. add     - Добавить лицо через камеру")
        print("2. file    - Добавить лицо из файла")
        print("3. test    - Тест распознавания в реальном времени")
        print("4. list    - Показать базу данных")
        print("5. exit    - Выход")
        
        choice = input("\nВаш выбор: ").strip().lower()
        
        if choice in ["1", "add"]:
            test_add_face_from_camera()
        elif choice in ["2", "file"]:
            test_add_from_file()
        elif choice in ["3", "test"]:
            test_recognition_live()
        elif choice in ["4", "list"]:
            show_database()
        elif choice in ["5", "exit", "q"]:
            print("\nВыход...")
            break
        else:
            print("Неизвестная команда")


if __name__ == "__main__":
    main()
