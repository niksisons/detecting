"""
База данных лиц с использованием InsightFace
InsightFace — современная библиотека для детекции и распознавания лиц

Преимущества:
- Очень быстрая детекция лиц
- Высокая точность распознавания  
- Работает на GPU (CUDA)
- Лёгкая интеграция
"""
import os
import pickle
from pathlib import Path
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import config

# Подавляем предупреждения ONNX
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("InsightFace не установлен. Установите: pip install insightface onnxruntime-gpu")


class FaceDatabase:
    """Класс для работы с базой данных лиц через InsightFace"""
    
    def __init__(self, db_path: Path = None, det_size: Tuple[int, int] = (640, 640)):
        """
        Инициализация базы данных лиц
        
        Args:
            db_path: Путь к папке базы данных
            det_size: Размер изображения для детекции (ширина, высота)
        """
        self.db_path = db_path or config.FACES_DB_DIR
        self.encodings_file = self.db_path / "face_encodings_insightface.pkl"
        self.face_encodings: Dict[str, List[np.ndarray]] = {}
        self.det_size = det_size
        
        # Создаём папку если не существует
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Инициализация InsightFace
        self.app = None
        if INSIGHTFACE_AVAILABLE:
            self._init_insightface()
        
        # Загружаем базу данных
        self.load_database()
    
    def _init_insightface(self):
        """Инициализация модели InsightFace"""
        try:
            print("Загрузка модели InsightFace...")
            
            # Создаём анализатор лиц
            # buffalo_l - хороший баланс скорости и точности
            # buffalo_sc - маленькая и быстрая модель
            self.app = FaceAnalysis(
                name='buffalo_l',  # или 'buffalo_sc' для скорости
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            
            # Подготавливаем модель с размером детекции
            self.app.prepare(ctx_id=0, det_size=self.det_size)
            
            print(f"InsightFace загружен успешно (det_size={self.det_size})")
            
        except Exception as e:
            print(f"Ошибка инициализации InsightFace: {e}")
            print("Попробуйте: pip install insightface onnxruntime-gpu")
            self.app = None
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Детекция всех лиц на кадре
        
        Args:
            frame: BGR изображение (OpenCV формат)
            
        Returns:
            Список словарей с информацией о лицах:
            - bbox: (x1, y1, x2, y2)
            - embedding: вектор эмбеддинга лица
            - det_score: уверенность детекции
            - landmarks: ключевые точки лица
        """
        if self.app is None:
            return []
        
        try:
            # InsightFace ожидает BGR изображение
            faces = self.app.get(frame)
            
            results = []
            for face in faces:
                # Получаем bbox
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                
                results.append({
                    "bbox": (x1, y1, x2, y2),
                    "embedding": face.embedding,
                    "det_score": float(face.det_score),
                    "landmarks": face.kps if hasattr(face, 'kps') else None,
                    "age": int(face.age) if hasattr(face, 'age') else None,
                    "gender": 'M' if hasattr(face, 'gender') and face.gender == 1 else 'F' if hasattr(face, 'gender') else None
                })
            
            return results
            
        except Exception as e:
            # Ошибка детекции - возвращаем пустой список
            return []
    
    def extract_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """
        Извлечь эмбеддинг лица из файла изображения
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            Вектор эмбеддинга или None если лицо не найдено
        """
        if self.app is None:
            print("InsightFace не инициализирован")
            return None
        
        try:
            # Читаем изображение
            img = cv2.imread(image_path)
            if img is None:
                print(f"Не удалось прочитать изображение: {image_path}")
                return None
            
            # Детектируем лица
            faces = self.detect_faces(img)
            
            if not faces:
                print(f"Лицо не найдено в {image_path}")
                return None
            
            # Берём первое (самое большое) лицо
            # Сортируем по площади bbox
            faces.sort(key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]), reverse=True)
            
            return faces[0]["embedding"]
            
        except Exception as e:
            print(f"Ошибка извлечения эмбеддинга: {e}")
            return None
    
    def add_person(self, name: str, image_path: str) -> bool:
        """
        Добавить человека в базу данных
        
        Args:
            name: Имя человека
            image_path: Путь к фото
            
        Returns:
            True если успешно
        """
        print(f"Добавление {name}...")
        
        embedding = self.extract_embedding(image_path)
        
        if embedding is not None:
            self.face_encodings.setdefault(name, []).append(embedding)
            print(f"  {name} добавлен успешно (эмбеддинг: {len(embedding)} измерений)")
            return True
        else:
            print(f"  Не удалось добавить {name}")
            return False
    
    def add_person_from_folder(self, name: str, folder_path: str) -> int:
        """
        Добавить несколько фото одного человека из папки
        
        Args:
            name: Имя человека
            folder_path: Путь к папке с фото
            
        Returns:
            Количество успешно добавленных фото
        """
        folder = Path(folder_path)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        
        added_count = 0
        for img_file in folder.iterdir():
            if img_file.suffix.lower() in image_extensions:
                if self.add_person(name, str(img_file)):
                    added_count += 1
        
        print(f"Добавлено {added_count} фото для {name}")
        return added_count
    
    def recognize_face(self, face_embedding: np.ndarray, threshold: float = None) -> Tuple[str, float]:
        """
        Распознать лицо по эмбеддингу
        
        Args:
            face_embedding: Вектор эмбеддинга лица
            threshold: Порог распознавания (меньше = строже)
            
        Returns:
            (Имя или "Unknown", дистанция)
        """
        if threshold is None:
            threshold = config.FACE_RECOGNITION_TOLERANCE
        
        if not self.face_encodings:
            return "Unknown", 1.0
        
        # Нормализуем входной эмбеддинг
        face_embedding = face_embedding / np.linalg.norm(face_embedding)
        
        min_distance = float('inf')
        recognized_name = "Unknown"
        
        for name, encodings_list in self.face_encodings.items():
            for known_embedding in encodings_list:
                # Нормализуем известный эмбеддинг
                known_norm = known_embedding / np.linalg.norm(known_embedding)
                
                # Косинусное расстояние (1 - косинусное сходство)
                cosine_sim = np.dot(face_embedding, known_norm)
                distance = 1 - cosine_sim
                
                if distance < min_distance:
                    min_distance = distance
                    if distance < threshold:
                        recognized_name = name
        
        return recognized_name, min_distance
    
    def recognize_faces_in_frame(self, frame: np.ndarray, threshold: float = None) -> List[Dict]:
        """
        Детектировать и распознать все лица на кадре
        
        Args:
            frame: BGR изображение
            threshold: Порог распознавания
            
        Returns:
            Список словарей с информацией о распознанных лицах
        """
        faces = self.detect_faces(frame)
        
        results = []
        for face in faces:
            name, distance = self.recognize_face(face["embedding"], threshold)
            
            results.append({
                "name": name,
                "distance": distance,
                "confidence": 1 - distance,
                "bbox": face["bbox"],
                "embedding": face["embedding"],
                "det_score": face["det_score"],
                "quality": (face["bbox"][2] - face["bbox"][0]) * (face["bbox"][3] - face["bbox"][1])  # Площадь как качество
            })
        
        return results
    
    def save_database(self):
        """Сохранить базу данных на диск"""
        with open(self.encodings_file, 'wb') as f:
            pickle.dump({
                'encodings': self.face_encodings,
                'model': 'insightface_buffalo_l'
            }, f)
        print(f"База данных сохранена: {self.encodings_file}")
    
    def load_database(self):
        """Загрузить базу данных с диска"""
        if self.encodings_file.exists():
            try:
                with open(self.encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, dict) and 'encodings' in data:
                        self.face_encodings = data['encodings']
                    else:
                        # Старый формат
                        self.face_encodings = data
                print(f"База данных загружена: {len(self.face_encodings)} человек")
            except Exception as e:
                print(f"Ошибка загрузки базы данных: {e}")
                self.face_encodings = {}
        else:
            print("База данных пуста, создана новая")
            self.face_encodings = {}
    
    def list_persons(self) -> List[str]:
        """Получить список всех людей в базе"""
        return list(self.face_encodings.keys())
    
    def remove_person(self, name: str) -> bool:
        """Удалить человека из базы"""
        if name in self.face_encodings:
            del self.face_encodings[name]
            print(f"{name} удалён из базы данных")
            return True
        return False
    
    def get_person_count(self, name: str) -> int:
        """Получить количество эмбеддингов для человека"""
        return len(self.face_encodings.get(name, []))


def main():
    """Демонстрация работы с базой данных лиц InsightFace"""
    print("=" * 60)
    print("УПРАВЛЕНИЕ БАЗОЙ ДАННЫХ ЛИЦ (InsightFace)")
    print("=" * 60)
    
    if not INSIGHTFACE_AVAILABLE:
        print("\nInsightFace не установлен!")
        print("Установите: pip install insightface onnxruntime-gpu")
        return
    
    db = FaceDatabase()
    
    print("\nДоступные команды:")
    print("1. add      - Добавить человека (одно фото)")
    print("2. folder   - Добавить человека (папка с фото)")
    print("3. list     - Показать всех людей в базе")
    print("4. remove   - Удалить человека")
    print("5. test     - Тест распознавания через камеру")
    print("6. exit     - Выход")
    
    while True:
        command = input("\nВведите команду: ").strip().lower()
        
        if command == "add":
            name = input("Имя человека: ").strip()
            image_path = input("Путь к фото: ").strip()
            if db.add_person(name, image_path):
                db.save_database()
        
        elif command == "folder":
            name = input("Имя человека: ").strip()
            folder_path = input("Путь к папке: ").strip()
            if db.add_person_from_folder(name, folder_path) > 0:
                db.save_database()
        
        elif command == "list":
            persons = db.list_persons()
            print(f"\nВсего в базе: {len(persons)} человек")
            for i, person in enumerate(persons, 1):
                count = db.get_person_count(person)
                print(f"{i}. {person} ({count} эмбеддингов)")
        
        elif command == "remove":
            name = input("Имя для удаления: ").strip()
            if db.remove_person(name):
                db.save_database()
        
        elif command == "test":
            test_camera_recognition(db)
        
        elif command == "exit":
            break
        
        else:
            print("Неизвестная команда")


def test_camera_recognition(db: FaceDatabase):
    """Тест распознавания лиц через веб-камеру"""
    print("\nТест распознавания через камеру")
    print("Нажмите Q для выхода, S для скриншота")
    print("-" * 40)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть камеру")
        return
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Распознаём каждый 3-й кадр для скорости
        if frame_count % 3 == 0:
            faces = db.recognize_faces_in_frame(frame)
            
            for face in faces:
                x1, y1, x2, y2 = face["bbox"]
                name = face["name"]
                conf = face["confidence"]
                
                # Цвет рамки: зелёный если распознан, красный если нет
                if name != "Unknown":
                    color = (0, 255, 0)
                    label = f"{name} ({conf:.0%})"
                else:
                    color = (0, 0, 255)
                    label = f"Unknown ({face['distance']:.2f})"
                
                # Рисуем рамку и метку
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(frame, (x1, y1 - 30), (x2, y1), color, -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Показываем кадр
        cv2.imshow("InsightFace Test - Q to exit", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("face_screenshot.jpg", frame)
            print("Скриншот сохранён: face_screenshot.jpg")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Тест завершён")


if __name__ == "__main__":
    main()
