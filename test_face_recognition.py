"""
Тестовый скрипт для проверки распознавания лиц
"""
import cv2
import face_recognition
from face_database import FaceDatabase
import config

def test_face_detection():
    """Тест детекции лиц через веб-камеру"""
    print("=" * 60)
    print("ТЕСТ РАСПОЗНАВАНИЯ ЛИЦ")
    print("=" * 60)
    
    # Загружаем базу данных
    db = FaceDatabase()
    print(f"\nЛюди в базе: {db.list_persons()}")
    print(f"Порог распознавания: {config.FACE_RECOGNITION_TOLERANCE}")
    print(f"Модель детекции: {config.FACE_DETECTION_MODEL}")
    
    # Открываем камеру
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Ошибка: не удалось открыть камеру!")
        return
    
    print("\nНажмите Q для выхода, S для сохранения скриншота")
    print("-" * 60)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Детектируем лица каждые 5 кадров
        if frame_count % 5 == 0:
            # Конвертируем в RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Уменьшаем для ускорения
            small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)
            
            # Детекция лиц
            face_locations = face_recognition.face_locations(small_frame, model=config.FACE_DETECTION_MODEL)
            
            # Масштабируем обратно
            face_locations = [(int(top*2), int(right*2), int(bottom*2), int(left*2)) 
                             for top, right, bottom, left in face_locations]
            
            if face_locations:
                # Получаем эмбеддинги
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    # Распознаём
                    name, distance = db.recognize_face(face_encoding)
                    
                    # Цвет рамки
                    if name != "Unknown":
                        color = (0, 255, 0)  # Зелёный - распознан
                        confidence = 1 - distance
                        label = f"{name} ({confidence:.0%})"
                    else:
                        color = (0, 0, 255)  # Красный - не распознан
                        label = f"Unknown (dist: {distance:.2f})"
                    
                    # Рисуем рамку
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    
                    # Рисуем текст
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                    cv2.putText(frame, label, (left + 6, bottom - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    print(f"Лицо найдено: {name}, дистанция: {distance:.3f}, порог: {config.FACE_RECOGNITION_TOLERANCE}")
            else:
                # Нет лиц
                cv2.putText(frame, "No faces detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Показываем кадр
        cv2.imshow("Face Recognition Test - Press Q to exit", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("face_test_screenshot.jpg", frame)
            print("Скриншот сохранён: face_test_screenshot.jpg")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nТест завершён")

if __name__ == "__main__":
    test_face_detection()
