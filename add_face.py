#!/usr/bin/env python3
"""
Скрипт для добавления лица в базу данных с веб-камеры
Использование: python add_face.py "Имя"
"""
import cv2
import sys
from pathlib import Path
from face_database import FaceDatabase
import config


def capture_face(name: str, num_photos: int = 3):
    """
    Захват фото лица с веб-камеры
    
    Args:
        name: Имя человека
        num_photos: Количество фото для захвата
    """
    print(f"\n📸 Добавление лица: {name}")
    print("=" * 50)
    
    # Создаём папку для фото
    person_dir = config.FACES_DB_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)
    
    # Открываем камеру
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Не удалось открыть веб-камеру!")
        return False
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print(f"\n📷 Нужно сделать {num_photos} фото")
    print("👉 Нажмите ПРОБЕЛ чтобы сделать фото")
    print("👉 Нажмите Q или ESC чтобы выйти")
    print("\n💡 Совет: делайте фото с разных ракурсов для лучшего распознавания\n")
    
    photos_taken = 0
    
    while photos_taken < num_photos:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Зеркальное отражение для удобства
        frame = cv2.flip(frame, 1)
        
        # Инструкции на кадре
        cv2.putText(frame, f"Photo {photos_taken + 1}/{num_photos}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "SPACE - capture, Q - quit", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Рамка по центру для позиционирования лица
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        size = 150
        cv2.rectangle(frame, (cx - size, cy - size), (cx + size, cy + size), (0, 255, 0), 2)
        cv2.putText(frame, "Position face here", 
                    (cx - 80, cy - size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow(f"Add Face: {name}", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Пробел - сделать фото
            # Сохраняем без рамки и текста
            ret, clean_frame = cap.read()
            if ret:
                clean_frame = cv2.flip(clean_frame, 1)
                photo_path = person_dir / f"{name}_{photos_taken + 1}.jpg"
                cv2.imwrite(str(photo_path), clean_frame)
                photos_taken += 1
                print(f"✅ Фото {photos_taken} сохранено: {photo_path}")
                
                # Визуальный фидбек
                cv2.putText(frame, "CAPTURED!", (w//2 - 80, h//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.imshow(f"Add Face: {name}", frame)
                cv2.waitKey(500)
        
        elif key in [ord('q'), ord('Q'), 27]:  # Q или ESC - выход
            print("\n⚠️ Отменено пользователем")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if photos_taken == 0:
        print("❌ Фото не сделаны")
        return False
    
    # Добавляем в базу данных
    print(f"\n🔄 Добавление {photos_taken} фото в базу данных...")
    db = FaceDatabase()
    
    added = 0
    for photo_path in person_dir.glob("*.jpg"):
        if db.add_person(name, str(photo_path)):
            added += 1
    
    if added > 0:
        db.save_database()
        print(f"\n✅ {name} успешно добавлен в базу! ({added} фото)")
        print(f"📁 Фото сохранены в: {person_dir}")
        return True
    else:
        print("❌ Не удалось добавить лицо в базу")
        return False


def add_from_file(name: str, image_path: str):
    """Добавить лицо из файла"""
    print(f"\n📸 Добавление лица из файла: {image_path}")
    
    db = FaceDatabase()
    if db.add_person(name, image_path):
        db.save_database()
        print(f"✅ {name} успешно добавлен!")
        return True
    return False


def list_database():
    """Показать всех людей в базе"""
    db = FaceDatabase()
    persons = db.list_persons()
    
    print("\n👥 База данных лиц:")
    print("=" * 40)
    
    if not persons:
        print("   (пусто)")
    else:
        for i, name in enumerate(persons, 1):
            count = len(db.face_encodings[name])
            print(f"   {i}. {name} ({count} фото)")
    
    print("=" * 40)
    print(f"Всего: {len(persons)} человек")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование:")
        print("  python add_face.py <имя>           - добавить с веб-камеры")
        print("  python add_face.py <имя> <фото>    - добавить из файла")
        print("  python add_face.py --list          - показать базу")
        sys.exit(1)
    
    if sys.argv[1] == "--list":
        list_database()
    elif len(sys.argv) == 2:
        name = sys.argv[1]
        capture_face(name)
    else:
        name = sys.argv[1]
        image_path = sys.argv[2]
        add_from_file(name, image_path)
