"""
Main module for discipline violation detection
Supports: webcam, video files, IP cameras, batch processing
"""
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import os

# Suppress TensorFlow/CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from ultralytics import YOLO
import json
from collections import defaultdict
import argparse
import config
from tqdm import tqdm

# Распознавание лиц (face_recognition/dlib)
FACE_BACKEND = "none"

try:
    from face_database import FaceDatabase
    import face_recognition
    FACE_BACKEND = "dlib"
    print("Используется face_recognition (dlib) для распознавания лиц")
except ImportError:
    # Fallback - создаём заглушку
    print("ВНИМАНИЕ: face_recognition не установлен. Распознавание лиц недоступно!")
    
    class FaceDatabase:
        def __init__(self, *args, **kwargs):
            self.face_encodings = {}
        def detect_faces(self, frame):
            return []
        def recognize_faces_in_frame(self, frame, threshold=None):
            return []
        def recognize_face(self, embedding, threshold=None):
            return "Unknown", 1.0
        def list_persons(self):
            return []
        def save_database(self):
            pass
        def load_database(self):
            pass


class ViolationDetector:
    """Class for discipline violation detection"""
    
    def __init__(self, model_path: str = None, source: str = "0", save_output: bool = True):
        """
        Initialize detector
        
        Args:
            model_path: Path to trained YOLO model
            source: Video source (0 for webcam, file path, folder, or IP camera)
            save_output: Whether to save processed video
        """
        # Use default model if not specified
        if model_path is None:
            model_path = str(config.YOLO_TRAINED_MODEL)
        
        self.model_path = model_path
        self.source = source
        self.save_output = save_output
        
        # Check model exists
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}\n"
                                   f"   Train model first: python train_model.py")
        
        # Load model
        print(f"Loading model: {self.model_path}")
        self.model = YOLO(self.model_path)
        
        # Face database
        self.face_db = FaceDatabase()
        
        # Violation tracking
        self.violations: Dict[int, Dict] = {}  # track_id -> violation_info
        self.violation_counters = defaultdict(int)  # track_id -> frame_count
        
        # Statistics
        self.total_violations = []
        self.current_recordings = {}  # track_id -> VideoWriter
        
        # Хранилище лучших лиц для каждого трека нарушения
        # track_id -> {"face_img": np.ndarray, "encoding": np.ndarray, "quality": float, "name": str}
        self.best_faces: Dict[int, Dict] = {}
        
        # Счётчик кадров для пропуска распознавания лиц
        self.face_frame_counter = 0
        
        # Video stream
        self.cap = None
        self.fps = config.FPS
        self.frame_width = 0
        self.frame_height = 0
        self.total_frames = 0
        
        # Output video
        self.output_writer = None
        
    def initialize_video_source(self):
        """Initialize video source"""
        print(f"Opening video stream: {self.source}")
        
        # Convert "0" to integer for webcam
        if self.source.isdigit():
            source = int(self.source)
        else:
            source = self.source
            # Check file exists
            if not source.startswith(('http://', 'https://', 'rtsp://')):
                video_path = Path(source)
                if not video_path.exists():
                    raise FileNotFoundError(f"Video file not found: {source}")
                print(f"Found video file: {video_path.name}")
        
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video stream: {self.source}")
        
        # Get video parameters
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or config.FPS
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video stream opened: {self.frame_width}x{self.frame_height} @ {self.fps} FPS")
        if self.total_frames > 0:
            duration = self.total_frames / self.fps
            print(f"Video duration: {duration:.1f} sec ({self.total_frames} frames)")
        
        # Create output video
        if self.save_output and not isinstance(source, int):
            output_filename = f"processed_{Path(self.source).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            output_path = config.OUTPUT_DIR / "processed_videos" / output_filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.output_writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                self.fps,
                (self.frame_width, self.frame_height)
            )
            print(f"Processed video will be saved to: {output_path}")
    
    def detect_and_recognize_faces(self, frame: np.ndarray, 
                                   violation_boxes: List[Tuple]) -> List[Dict]:
        """
        Detect and recognize faces in frame
        
        Args:
            frame: Frame to analyze
            violation_boxes: List of violation bboxes [(track_id, x1, y1, x2, y2), ...]
            
        Returns:
            List[Dict]: List of recognized faces linked to violations
        """
        recognized_faces = []
        
        # Используем универсальный метод recognize_faces_in_frame
        # который есть во всех версиях FaceDatabase
        try:
            if hasattr(self.face_db, 'recognize_faces_in_frame'):
                # Унифицированный интерфейс FaceDatabase
                faces = self.face_db.recognize_faces_in_frame(frame)
                
                for face in faces:
                    bbox = face["bbox"]
                    left, top, right, bottom = bbox
                    
                    face_center_x = (left + right) // 2
                    face_center_y = (top + bottom) // 2
                    
                    # Найти ближайшее нарушение
                    nearest_track_id = None
                    min_dist_to_violation = float('inf')
                    
                    for track_id, vx1, vy1, vx2, vy2 in violation_boxes:
                        violation_center_x = (vx1 + vx2) // 2
                        violation_center_y = (vy1 + vy2) // 2
                        
                        dist = np.sqrt((face_center_x - violation_center_x)**2 + 
                                      (face_center_y - violation_center_y)**2)
                        
                        max_allowed_dist = max(self.frame_width, self.frame_height) * 0.5
                        
                        if dist < min_dist_to_violation and dist < max_allowed_dist:
                            min_dist_to_violation = dist
                            nearest_track_id = track_id
                    
                    recognized_faces.append({
                        "name": face["name"],
                        "distance": face["distance"],
                        "bbox": bbox,
                        "encoding": face.get("embedding"),
                        "quality": face.get("quality", (right-left) * (bottom-top)),
                        "nearest_track_id": nearest_track_id
                    })
            
            elif FACE_BACKEND == "dlib":
                # Старый интерфейс face_recognition (dlib)
                import face_recognition
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Масштабирование для скорости
                scale = 1.0
                small_frame = rgb_frame
                if max(rgb_frame.shape[:2]) > 800:
                    scale = 800.0 / max(rgb_frame.shape[:2])
                    small_frame = cv2.resize(rgb_frame, None, fx=scale, fy=scale)
                
                face_locations = face_recognition.face_locations(small_frame, 
                                                                 model=config.FACE_DETECTION_MODEL)
                
                if face_locations:
                    if scale != 1.0:
                        face_locations = [(int(top/scale), int(right/scale), 
                                          int(bottom/scale), int(left/scale)) 
                                         for top, right, bottom, left in face_locations]
                    
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    
                    for face_location, face_encoding in zip(face_locations, face_encodings):
                        top, right, bottom, left = face_location
                        
                        name, distance = self.face_db.recognize_face(face_encoding)
                        
                        face_center_x = (left + right) // 2
                        face_center_y = (top + bottom) // 2
                        
                        nearest_track_id = None
                        min_dist_to_violation = float('inf')
                        
                        for track_id, vx1, vy1, vx2, vy2 in violation_boxes:
                            violation_center_x = (vx1 + vx2) // 2
                            violation_center_y = (vy1 + vy2) // 2
                            
                            dist = np.sqrt((face_center_x - violation_center_x)**2 + 
                                          (face_center_y - violation_center_y)**2)
                            
                            max_allowed_dist = max(self.frame_width, self.frame_height) * 0.5
                            
                            if dist < min_dist_to_violation and dist < max_allowed_dist:
                                min_dist_to_violation = dist
                                nearest_track_id = track_id
                        
                        recognized_faces.append({
                            "name": name,
                            "distance": distance,
                            "bbox": (left, top, right, bottom),
                            "encoding": face_encoding,
                            "quality": (right-left) * (bottom-top),
                            "nearest_track_id": nearest_track_id
                        })
                        
        except Exception as e:
            # Ошибка детекции - возвращаем пустой список
            pass
        
        return recognized_faces
    
    def update_best_face(self, track_id: int, face_data: Dict, frame: np.ndarray):
        """
        Обновить лучшее лицо для нарушения если текущее качественнее
        
        Args:
            track_id: ID трека нарушения
            face_data: Данные о лице
            frame: Текущий кадр
        """
        current_quality = face_data["quality"]
        
        # Если это первое лицо или оно лучше предыдущего
        if track_id not in self.best_faces or current_quality > self.best_faces[track_id]["quality"]:
            left, top, right, bottom = face_data["bbox"]
            
            # Вырезаем лицо с отступом
            margin = 30
            face_img = frame[max(0, top-margin):min(frame.shape[0], bottom+margin), 
                            max(0, left-margin):min(frame.shape[1], right+margin)].copy()
            
            self.best_faces[track_id] = {
                "face_img": face_img,
                "encoding": face_data["encoding"],
                "quality": current_quality,
                "name": face_data["name"],
                "distance": face_data["distance"],
                "bbox": face_data["bbox"]
            }
            
            # Обновляем имя нарушителя если распознан
            if track_id in self.violations:
                self.violations[track_id]["person_name"] = face_data["name"]
            
            print(f"   Лучшее лицо обновлено для трека {track_id}: {face_data['name']} (качество: {current_quality})")
    
    def save_face_image(self, frame: np.ndarray, bbox: Tuple, 
                       violation_id: str) -> str:
        """Save face image"""
        left, top, right, bottom = bbox
        
        # Crop face with margin
        margin = 20
        face_img = frame[max(0, top-margin):bottom+margin, 
                        max(0, left-margin):right+margin]
        
        # Path to save
        faces_dir = config.OUTPUT_DIR / "faces"
        faces_dir.mkdir(exist_ok=True)
        
        face_path = faces_dir / f"{violation_id}_face.jpg"
        cv2.imwrite(str(face_path), face_img)
        
        return str(face_path)
    
    def start_recording(self, track_id: int, violation_type: str) -> cv2.VideoWriter:
        """Start recording violation video"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"{violation_type}_{track_id}_{timestamp}{config.VIDEO_EXTENSION}"
        video_path = config.OUTPUT_DIR / "videos" / video_filename
        video_path.parent.mkdir(exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_CODEC)
        writer = cv2.VideoWriter(
            str(video_path),
            fourcc,
            self.fps,
            (self.frame_width, self.frame_height)
        )
        
        return writer, str(video_path)
    
    def process_frame(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        """
        Обработка одного кадра
        
        Args:
            frame: Исходный кадр
            frame_number: Номер кадра
            
        Returns:
            np.ndarray: Обработанный кадр с аннотациями
        """
        # Детекция объектов с трекингом
        results = self.model.track(frame, 
                                   conf=config.YOLO_CONFIDENCE,
                                   iou=config.YOLO_IOU,
                                   persist=True,
                                   tracker="bytetrack.yaml")
        
        annotated_frame = frame.copy()
        current_time = datetime.now()
        
        active_tracks = set()
        violation_boxes = []  # Список (track_id, x1, y1, x2, y2) для привязки лиц
        
        # Обработка детекций
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            
            for box in boxes:
                # Получаем данные
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                track_id = int(box.id[0]) if box.id is not None else -1
                
                if track_id == -1:
                    continue
                
                active_tracks.add(track_id)
                x1, y1, x2, y2 = map(int, xyxy)
                
                # Тип нарушения
                violation_type = config.VIOLATION_CLASSES.get(cls, "unknown")
                
                # Увеличиваем счётчик кадров для этого трека
                self.violation_counters[track_id] += 1
                
                # Проверяем достигнут ли порог для записи
                frames_threshold = config.VIOLATION_DURATION_THRESHOLD * self.fps
                
                if self.violation_counters[track_id] >= frames_threshold:
                    # Начинаем запись если ещё не начали
                    if track_id not in self.violations:
                        print(f"Нарушение обнаружено: {violation_type} (ID: {track_id})")
                        
                        writer, video_path = self.start_recording(track_id, violation_type)
                        
                        # Проверяем, есть ли уже сохранённое лицо для этого трека
                        person_name = "Unknown"
                        if track_id in self.best_faces:
                            person_name = self.best_faces[track_id]["name"]
                        
                        self.violations[track_id] = {
                            "type": violation_type,
                            "start_time": current_time,
                            "end_time": None,
                            "video_path": video_path,
                            "face_path": None,
                            "person_name": person_name,
                            "confidence": conf,
                            "frames": []
                        }
                        
                        self.current_recordings[track_id] = writer
                    
                    # Обновляем информацию
                    self.violations[track_id]["end_time"] = current_time
                    self.violations[track_id]["frames"].append(frame_number)
                    
                    # Добавляем в список для привязки лиц
                    violation_boxes.append((track_id, x1, y1, x2, y2))
                
                # Отрисовка бокса
                color = config.COLORS.get(violation_type, (255, 255, 255))
                
                # Толстая рамка для нарушений
                thickness = 3 if track_id in self.violations else 2
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                
                # Метка с типом нарушения
                violation_label = config.VIOLATION_LABELS.get(violation_type, violation_type)
                label = f"{violation_label} {conf:.2f}"
                
                if track_id in self.violations:
                    person_name = self.violations[track_id]["person_name"]
                    duration = (current_time - self.violations[track_id]["start_time"]).seconds
                    label = f"{violation_label} | {person_name} | {duration}s"
                
                # Рисуем фон текста
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(annotated_frame, 
                            (x1, y1 - text_height - 10), 
                            (x1 + text_width + 10, y1), 
                            color, -1)
                
                # Рисуем текст
                cv2.putText(annotated_frame, label, (x1 + 5, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Записываем кадр в видео
                if track_id in self.current_recordings:
                    self.current_recordings[track_id].write(annotated_frame)
        
        # === ДЕТЕКЦИЯ И РАСПОЗНАВАНИЕ ЛИЦ ===
        # Детектируем лица каждые N кадров для оптимизации
        self.face_frame_counter += 1
        should_detect_faces = (self.face_frame_counter % config.FACE_RECOGNITION_FRAME_SKIP == 0)
        
        if should_detect_faces and violation_boxes:
            faces = self.detect_and_recognize_faces(frame, violation_boxes)
            
            for face in faces:
                # Рисуем рамку лица
                left, top, right, bottom = face["bbox"]
                face_color = config.COLORS["face"]
                
                # Зелёная рамка если распознан, синяя если нет
                if face["name"] != "Unknown":
                    face_color = (0, 255, 0)  # Зелёный
                
                cv2.rectangle(annotated_frame, (left, top), (right, bottom), face_color, 2)
                
                # Метка лица
                name_label = f"{face['name']}"
                if face["name"] != "Unknown":
                    name_label += f" ({1 - face['distance']:.0%})"
                
                # Фон для текста
                (tw, th), _ = cv2.getTextSize(name_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(annotated_frame, (left, top - th - 10), (left + tw + 10, top), face_color, -1)
                cv2.putText(annotated_frame, name_label, (left + 5, top - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Привязываем лицо к ближайшему нарушению
                nearest_track_id = face["nearest_track_id"]
                
                if nearest_track_id is not None:
                    # Обновляем лучшее лицо для этого трека
                    self.update_best_face(nearest_track_id, face, frame)
        
        # Завершение записи для пропавших треков
        disappeared_tracks = set(self.violations.keys()) - active_tracks
        
        for track_id in disappeared_tracks:
            # Продолжаем запись во время буфера
            if track_id in self.current_recordings:
                buffer_frames = config.BUFFER_AFTER_VIOLATION * self.fps
                
                # Проверяем, прошло ли достаточно времени
                last_frame = self.violations[track_id]["frames"][-1] if self.violations[track_id]["frames"] else 0
                
                if frame_number - last_frame > buffer_frames:
                    # Завершаем запись
                    self.current_recordings[track_id].release()
                    del self.current_recordings[track_id]
                    
                    # Сохраняем лучшее лицо для этого нарушения
                    if track_id in self.best_faces:
                        best_face = self.best_faces[track_id]
                        violation_id = f"{track_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        
                        # Сохраняем изображение лучшего лица
                        faces_dir = config.OUTPUT_DIR / "faces"
                        faces_dir.mkdir(exist_ok=True)
                        face_path = faces_dir / f"{violation_id}_face.jpg"
                        
                        if best_face["face_img"] is not None and best_face["face_img"].size > 0:
                            cv2.imwrite(str(face_path), best_face["face_img"])
                            self.violations[track_id]["face_path"] = str(face_path)
                        
                        # Обновляем имя нарушителя
                        self.violations[track_id]["person_name"] = best_face["name"]
                        
                        print(f"   Лицо сохранено: {best_face['name']}")
                    
                    # Сохраняем нарушение в отчёт
                    self.total_violations.append(self.violations[track_id].copy())
                    
                    print(f"Запись завершена (ID: {track_id})")
                    print(f"   Файл: {self.violations[track_id]['video_path']}")
                    print(f"   Нарушитель: {self.violations[track_id]['person_name']}")
                else:
                    # Продолжаем запись
                    if track_id in self.current_recordings:
                        self.current_recordings[track_id].write(annotated_frame)
        
        # Information panel
        info_text = f"Frame: {frame_number} | Active: {len(self.violations)} | Total: {len(self.total_violations)}"
        cv2.rectangle(annotated_frame, (5, 5), (350, 35), (0, 0, 0), -1)
        cv2.putText(annotated_frame, info_text, (10, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return annotated_frame
    
    def generate_report(self):
        """Generate violations report"""
        print("\nGenerating report...")
        
        report_data = []
        
        for i, violation in enumerate(self.total_violations, 1):
            report_data.append({
                "id": i,
                "type": violation["type"],
                "start_time": violation["start_time"].isoformat(),
                "end_time": violation["end_time"].isoformat() if violation["end_time"] else None,
                "duration_seconds": (violation["end_time"] - violation["start_time"]).total_seconds() if violation["end_time"] else 0,
                "person_name": violation["person_name"],
                "confidence": violation["confidence"],
                "video_path": violation["video_path"],
                "face_path": violation["face_path"],
                "frames_count": len(violation["frames"])
            })
        
        # Save JSON
        report_path = config.OUTPUT_DIR / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        print(f"Report saved: {report_path}")
        
        # Statistics
        print("\nSTATISTICS:")
        print(f"   Total violations: {len(self.total_violations)}")
        
        # By type
        by_type = defaultdict(int)
        for v in self.total_violations:
            by_type[v["type"]] += 1
        
        print("\n   By type:")
        for vtype, count in by_type.items():
            print(f"     - {vtype}: {count}")
        
        # By person
        by_person = defaultdict(int)
        for v in self.total_violations:
            by_person[v["person_name"]] += 1
        
        print("\n   By violator:")
        for person, count in sorted(by_person.items(), key=lambda x: x[1], reverse=True):
            print(f"     - {person}: {count}")
        
        return report_path
    
    def run(self, show_video: bool = True):
        """
        Run detection
        
        Args:
            show_video: Show video window (can disable for files for speed)
        """
        try:
            self.initialize_video_source()
            
            print("\nStarting violation detection...")
            if show_video:
                print("   Press 'q' to exit")
            print()
            
            frame_number = 0
            frame_skip = getattr(config, 'FRAME_SKIP', 1)
            
            # Progress bar for video files
            progress_bar = None
            if self.total_frames > 0:
                progress_bar = tqdm(total=self.total_frames, desc="Processing", unit="frame")
            
            last_annotated_frame = None
            
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("\nVideo processing complete")
                    break
                
                frame_number += 1
                
                # Skip frames for speed
                if frame_number % frame_skip != 0:
                    # Use previous annotated frame for recording
                    if self.output_writer and last_annotated_frame is not None:
                        self.output_writer.write(last_annotated_frame)
                    if progress_bar:
                        progress_bar.update(1)
                    continue
                
                # Process frame
                annotated_frame = self.process_frame(frame, frame_number)
                last_annotated_frame = annotated_frame
                
                # Save processed frame
                if self.output_writer:
                    self.output_writer.write(annotated_frame)
                
                # Display (optional)
                if show_video:
                    try:
                        cv2.imshow("Discipline Monitor", annotated_frame)
                        
                        # Exit on 'q' press
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("\nStopped by user")
                            break
                    except cv2.error:
                        # GUI not supported, disable display
                        show_video = False
                
                # Update progress
                if progress_bar:
                    progress_bar.update(1)
            
            if progress_bar:
                progress_bar.close()
            
        except KeyboardInterrupt:
            print("\nStopped by Ctrl+C")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        print("\nCleaning up...")
        
        # End all recordings
        for writer in self.current_recordings.values():
            writer.release()
        
        # Close output video
        if self.output_writer:
            self.output_writer.release()
            print("Processed video saved")
        
        # Close video stream
        if self.cap:
            self.cap.release()
        
        # Safe window closing (may not work in headless mode)
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass  # Ignore error if GUI not supported
        
        # Generate report
        if self.total_violations:
            self.generate_report()
        
        print("Done")


def process_video_batch(video_folder: str, model_path: str = None, show_video: bool = False):
    """
    Batch process multiple videos
    
    Args:
        video_folder: Folder with video files
        model_path: Path to model
        show_video: Show video during processing
    """
    video_folder = Path(video_folder)
    
    if not video_folder.exists():
        raise FileNotFoundError(f"Folder not found: {video_folder}")
    
    # Find all video files
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_folder.glob(ext))
        video_files.extend(video_folder.glob(ext.upper()))
    
    if not video_files:
        print(f"No video files found in: {video_folder}")
        return
    
    print(f"\nFound video files: {len(video_files)}")
    print("=" * 70)
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Processing: {video_path.name}")
        print("-" * 70)
        
        try:
            detector = ViolationDetector(
                model_path=model_path, 
                source=str(video_path),
                save_output=True
            )
            detector.run(show_video=show_video)
            
        except Exception as e:
            print(f"Error processing {video_path.name}: {e}")
            continue
    
    print("\n" + "=" * 70)
    print("ALL VIDEOS PROCESSED!")
    print("=" * 70)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Discipline violation detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Webcam:
  python detect_violations.py --source 0

  # Video file:
  python detect_violations.py --source "video.mp4"

  # Video file without display (faster):
  python detect_violations.py --source "video.mp4" --no-display

  # Batch processing folder:
  python detect_violations.py --batch "videos_folder" --no-display

  # With custom model:
  python detect_violations.py --source "video.mp4" --model "models/custom.pt"
        """
    )
    
    # Input
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--source", type=str, default="0",
                            help="Video source (0 for webcam, file path or URL)")
    input_group.add_argument("--batch", type=str,
                            help="Folder with videos for batch processing")
    
    # Parameters
    parser.add_argument("--model", type=str, default=None,
                       help="Path to YOLO model (default: models/best.pt)")
    parser.add_argument("--no-display", action="store_true",
                       help="Don't show video during processing (faster for files)")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save processed video")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("DISCIPLINE MONITORING SYSTEM")
    print("=" * 70)
    
    # Batch processing
    if args.batch:
        process_video_batch(
            video_folder=args.batch,
            model_path=args.model,
            show_video=not args.no_display
        )
    else:
        # Normal single source processing
        detector = ViolationDetector(
            model_path=args.model, 
            source=args.source,
            save_output=not args.no_save
        )
        detector.run(show_video=not args.no_display)
    
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    print(f"\nResults in folder: {config.OUTPUT_DIR.absolute()}")
    print("   - processed_videos/ - full processed videos")
    print("   - videos/ - violation clips")
    print("   - faces/ - violator face images")
    print("   - report_*.json - JSON reports")


if __name__ == "__main__":
    main()