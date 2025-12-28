"""
База данных лиц с использованием ONNX моделей (без InsightFace)
Использует SCRFD для детекции и ArcFace для распознавания
Не требует Visual C++ Build Tools
"""
import os
import pickle
from pathlib import Path
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from urllib.request import urlretrieve
import config

# Подавляем предупреждения ONNX
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("onnxruntime не установлен. Установите: pip install onnxruntime-gpu")


class FaceDetectorONNX:
    """Детектор лиц на базе Haar Cascade (без внешних моделей)"""
    
    def __init__(self, models_dir: Path = None, det_size: Tuple[int, int] = (640, 640)):
        """
        Инициализация детектора
        
        Args:
            models_dir: Папка для хранения моделей (не используется)
            det_size: Размер изображения для детекции (не используется)
        """
        self.det_size = det_size
        
        # Используем встроенный Haar Cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        print("Детектор Haar Cascade загружен")
    
    def detect(self, img: np.ndarray, score_threshold: float = 0.5) -> List[Dict]:
        """
        Детекция лиц на изображении
        
        Args:
            img: BGR изображение
            score_threshold: Минимальный порог (не используется для Haar)
            
        Returns:
            Список словарей с bbox и score
        """
        results = []
        
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            for (x, y, w, h) in faces:
                results.append({
                    "bbox": (x, y, x + w, y + h),
                    "score": 0.99
                })
        except Exception as e:
            pass
        
        return results
    
    def get_embedding(self, img: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Получить эмбеддинг лица (HOG дескриптор)
        
        Args:
            img: Исходное изображение
            face_bbox: (x1, y1, x2, y2)
            
        Returns:
            HOG дескриптор
        """
        try:
            x1, y1, x2, y2 = face_bbox
            
            # Вырезаем лицо с отступом
            h, w = img.shape[:2]
            margin = int((x2 - x1) * 0.1)
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)
            
            face_img = img[y1:y2, x1:x2]
            
            if face_img.size == 0:
                return None
            
            # Resize для HOG (112x112)
            face_resized = cv2.resize(face_img, (112, 112))
            gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            
            # HOG дескриптор
            hog = cv2.HOGDescriptor((112, 112), (16, 16), (8, 8), (8, 8), 9)
            embedding = hog.compute(gray).flatten()
            
            # Нормализуем
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            return embedding
            
        except Exception as e:
            return None


class FaceDatabase:
    """Класс для работы с базой данных лиц через ONNX"""
    
    def __init__(self, db_path: Path = None):
        """
        Инициализация базы данных лиц
        
        Args:
            db_path: Путь к папке базы данных
        """
        self.db_path = db_path or config.FACES_DB_DIR
        self.encodings_file = self.db_path / "face_encodings_onnx.pkl"
        self.face_encodings: Dict[str, List[np.ndarray]] = {}
        
        # Создаём папку если не существует
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Инициализация детектора
        self.detector = None
        if ONNX_AVAILABLE:
            print("Инициализация детектора лиц ONNX...")
            self.detector = FaceDetectorONNX()
        
        # Загружаем базу данных
        self.load_database()
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Детекция лиц на кадре с использованием OpenCV DNN (fallback)
        
        Args:
            frame: BGR изображение
            
        Returns:
            Список словарей с информацией о лицах
        """
        results = []
        
        # Используем OpenCV DNN детектор как fallback
        # Он не требует дополнительных зависимостей
        try:
            # Caffe модель для детекции лиц
            model_file = self.db_path / "opencv_face_detector_uint8.pb"
            config_file = self.db_path / "opencv_face_detector.pbtxt"
            
            # Если файлы не существуют, используем Haar Cascade
            if not model_file.exists():
                return self._detect_faces_haar(frame)
            
            net = cv2.dnn.readNetFromTensorflow(str(model_file), str(config_file))
            
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
            net.setInput(blob)
            detections = net.forward()
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)
                    
                    results.append({
                        "bbox": (x1, y1, x2, y2),
                        "score": float(confidence)
                    })
        
        except Exception as e:
            # Fallback на Haar Cascade
            return self._detect_faces_haar(frame)
        
        return results
    
    def _detect_faces_haar(self, frame: np.ndarray) -> List[Dict]:
        """Детекция лиц с помощью Haar Cascade (всегда работает)"""
        results = []
        
        try:
            # Haar cascade встроен в OpenCV
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                results.append({
                    "bbox": (x, y, x + w, y + h),
                    "score": 0.99  # Haar не даёт score
                })
        
        except Exception as e:
            pass
        
        return results
    
    def get_face_embedding(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Получить эмбеддинг лица
        Использует простой метод на основе HOG + PCA
        
        Args:
            frame: Исходное изображение
            bbox: (x1, y1, x2, y2)
            
        Returns:
            Вектор эмбеддинга
        """
        try:
            x1, y1, x2, y2 = bbox
            
            # Вырезаем лицо
            face_img = frame[y1:y2, x1:x2]
            
            if face_img.size == 0:
                return None
            
            # Resize до стандартного размера
            face_resized = cv2.resize(face_img, (112, 112))
            
            # Конвертируем в grayscale
            gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            
            # Используем HOG дескриптор
            hog = cv2.HOGDescriptor((112, 112), (16, 16), (8, 8), (8, 8), 9)
            embedding = hog.compute(gray)
            
            # Нормализуем
            embedding = embedding.flatten()
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            return embedding
            
        except Exception as e:
            return None
    
    def extract_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """
        Извлечь эмбеддинг лица из файла
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            Вектор эмбеддинга или None
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Не удалось прочитать: {image_path}")
                return None
            
            faces = self.detect_faces(img)
            
            if not faces:
                print(f"Лицо не найдено в {image_path}")
                return None
            
            # Берём самое большое лицо
            faces.sort(key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]), reverse=True)
            
            best_face = faces[0]
            embedding = self.get_face_embedding(img, best_face["bbox"])
            
            return embedding
            
        except Exception as e:
            print(f"Ошибка: {e}")
            return None
    
    def add_person(self, name: str, image_path: str) -> bool:
        """Добавить человека в базу"""
        print(f"Добавление {name}...")
        
        embedding = self.extract_embedding(image_path)
        
        if embedding is not None:
            self.face_encodings.setdefault(name, []).append(embedding)
            print(f"  {name} добавлен успешно")
            return True
        else:
            print(f"  Не удалось добавить {name}")
            return False
    
    def add_person_from_folder(self, name: str, folder_path: str) -> int:
        """Добавить несколько фото одного человека"""
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
            face_embedding: Вектор эмбеддинга
            threshold: Порог распознавания
            
        Returns:
            (Имя или "Unknown", дистанция)
        """
        if threshold is None:
            threshold = config.FACE_RECOGNITION_TOLERANCE
        
        if not self.face_encodings:
            return "Unknown", 1.0
        
        # Нормализуем
        face_embedding = face_embedding / (np.linalg.norm(face_embedding) + 1e-8)
        
        min_distance = float('inf')
        recognized_name = "Unknown"
        
        for name, encodings_list in self.face_encodings.items():
            for known_embedding in encodings_list:
                known_norm = known_embedding / (np.linalg.norm(known_embedding) + 1e-8)
                
                # Косинусное расстояние
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
            embedding = self.get_face_embedding(frame, face["bbox"])
            
            if embedding is not None:
                name, distance = self.recognize_face(embedding, threshold)
                
                results.append({
                    "name": name,
                    "distance": distance,
                    "confidence": max(0, 1 - distance),
                    "bbox": face["bbox"],
                    "embedding": embedding,
                    "det_score": face["score"],
                    "quality": (face["bbox"][2] - face["bbox"][0]) * (face["bbox"][3] - face["bbox"][1])
                })
        
        return results
    
    def save_database(self):
        """Сохранить базу на диск"""
        with open(self.encodings_file, 'wb') as f:
            pickle.dump({
                'encodings': self.face_encodings,
                'model': 'opencv_hog'
            }, f)
        print(f"База данных сохранена: {self.encodings_file}")
    
    def load_database(self):
        """Загрузить базу с диска"""
        if self.encodings_file.exists():
            try:
                with open(self.encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, dict) and 'encodings' in data:
                        self.face_encodings = data['encodings']
                    else:
                        self.face_encodings = data
                print(f"База данных загружена: {len(self.face_encodings)} человек")
            except Exception as e:
                print(f"Ошибка загрузки: {e}")
                self.face_encodings = {}
        else:
            print("База данных пуста, создана новая")
            self.face_encodings = {}
    
    def list_persons(self) -> List[str]:
        """Получить список всех людей"""
        return list(self.face_encodings.keys())
    
    def remove_person(self, name: str) -> bool:
        """Удалить человека из базы"""
        if name in self.face_encodings:
            del self.face_encodings[name]
            print(f"{name} удалён")
            return True
        return False
    
    def get_person_count(self, name: str) -> int:
        """Количество эмбеддингов для человека"""
        return len(self.face_encodings.get(name, []))


def main():
    """Демо работы с базой данных лиц"""
    print("=" * 60)
    print("УПРАВЛЕНИЕ БАЗОЙ ДАННЫХ ЛИЦ (OpenCV)")
    print("=" * 60)
    
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
    """Тест распознавания через камеру"""
    print("\nТест распознавания через камеру")
    print("Нажмите Q для выхода")
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
        
        # Распознаём каждый 3-й кадр
        if frame_count % 3 == 0:
            faces = db.recognize_faces_in_frame(frame)
            
            for face in faces:
                x1, y1, x2, y2 = face["bbox"]
                name = face["name"]
                conf = face["confidence"]
                
                if name != "Unknown":
                    color = (0, 255, 0)
                    label = f"{name} ({conf:.0%})"
                else:
                    color = (0, 0, 255)
                    label = f"Unknown ({face['distance']:.2f})"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(frame, (x1, y1 - 30), (x2, y1), color, -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Face Recognition Test - Q to exit", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Тест завершён")


if __name__ == "__main__":
    main()
