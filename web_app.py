"""
🎯 Веб-интерфейс системы мониторинга дисциплины
Streamlit приложение с детекцией нарушений и распознаванием лиц
"""
import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import json
import tempfile
import os
from collections import defaultdict
import pandas as pd

# Настройка страницы (должна быть первой командой Streamlit)
st.set_page_config(
    page_title="Discipline Monitor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Подавляем предупреждения TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Импорт конфигурации и модулей проекта
import config
from ultralytics import YOLO

# Импорт модуля распознавания лиц (face_recognition/dlib)
FACE_RECOGNITION_AVAILABLE = False
FaceDatabase = None
FACE_BACKEND = "None"

try:
    from face_database import FaceDatabase
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
    FACE_BACKEND = "dlib"
except ImportError:
    print("ВНИМАНИЕ: face_recognition не установлен. Распознавание лиц недоступно.")


# ==================== СТИЛИ ====================
def apply_custom_styles():
    """Применить кастомные CSS стили"""
    st.markdown("""
    <style>
    /* Основной контейнер */
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        color: white;
        text-align: center;
    }
    
    /* Карточки статистики */
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 5px 0;
    }
    
    .stat-card h2 {
        font-size: 2em;
        margin: 0;
    }
    
    .stat-card p {
        margin: 5px 0 0 0;
        opacity: 0.8;
    }
    
    /* Карточка нарушения */
    .violation-card {
        background: #fff;
        border-left: 4px solid #e74c3c;
        padding: 10px;
        margin: 5px 0;
        border-radius: 0 10px 10px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* Статус индикаторы */
    .status-active {
        background: #27ae60;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        display: inline-block;
    }
    
    .status-inactive {
        background: #e74c3c;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        display: inline-block;
    }
    
    /* Улучшение боковой панели */
    .css-1d391kg {
        padding-top: 1rem;
    }
    
    /* Кнопки */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        padding: 10px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)


# ==================== ИНИЦИАЛИЗАЦИЯ СОСТОЯНИЯ ====================
def init_session_state():
    """Инициализация состояния сессии"""
    defaults = {
        'is_running': False,
        'violations_log': [],
        'total_violations': 0,
        'violations_by_type': defaultdict(int),
        'violations_by_person': defaultdict(int),
        'start_time': None,
        'fps': 0,
        'processed_frames': 0,
        'active_violations': {},
        'detected_faces': [],
        'face_db_instance': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ==================== ЗАГРУЗКА РЕСУРСОВ ====================
@st.cache_resource
def load_model(model_path: str):
    """Загрузка модели YOLO"""
    if not Path(model_path).exists():
        return None
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None


@st.cache_resource
def load_face_database():
    """Загрузка базы данных лиц"""
    if not FACE_RECOGNITION_AVAILABLE or FaceDatabase is None:
        return None
    try:
        return FaceDatabase()
    except Exception as e:
        st.warning(f"Ошибка загрузки базы лиц: {e}")
        return None


# ==================== РАСПОЗНАВАНИЕ ЛИЦ ====================
def detect_and_recognize_faces(frame: np.ndarray, face_db, threshold: float = 0.5):
    """
    Детекция и распознавание лиц на кадре
    
    Args:
        frame: BGR изображение
        face_db: База данных лиц
        threshold: Порог распознавания
        
    Returns:
        Список распознанных лиц с bbox и именами
    """
    if face_db is None:
        return []
    
    try:
        # Используем метод recognize_faces_in_frame если есть
        if hasattr(face_db, 'recognize_faces_in_frame'):
            return face_db.recognize_faces_in_frame(frame, threshold)
        
        # Fallback для альтернативных методов
        if hasattr(face_db, 'extract_embedding_from_frame'):
            results = []
            face_data = face_db.extract_embedding_from_frame(frame)
            
            for embedding, (x, y, w, h) in face_data:
                name, distance = face_db.recognize_face(embedding, threshold)
                results.append({
                    "name": name,
                    "distance": distance,
                    "confidence": max(0, 1 - distance),
                    "bbox": (x, y, x + w, y + h),
                    "embedding": embedding
                })
            return results
        
        return []
        
    except Exception as e:
        return []


def draw_face_annotations(frame: np.ndarray, faces: list) -> np.ndarray:
    """Отрисовка аннотаций лиц на кадре"""
    for face in faces:
        bbox = face.get("bbox", (0, 0, 0, 0))
        name = face.get("name", "Unknown")
        conf = face.get("confidence", 0)
        
        if len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Цвет: зелёный если распознан, красный если нет
            if name != "Unknown":
                color = (0, 255, 0)
                label = f"{name} ({conf:.0%})"
            else:
                color = (0, 0, 255)
                label = "Unknown"
            
            # Рамка лица
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Подпись
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - text_h - 8), (x1 + text_w + 5, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return frame


# ==================== БОКОВАЯ ПАНЕЛЬ ====================
def render_sidebar():
    """Боковая панель с настройками"""
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # Источник видео
        st.markdown("### 📹 Video Source")
        source_type = st.radio(
            "Select source:",
            ["📁 Video File", "📷 Webcam", "🌐 URL/RTSP"],
            key="source_type"
        )
        
        video_source = None
        uploaded_file = None
        
        if source_type == "📁 Video File":
            uploaded_file = st.file_uploader(
                "Upload video",
                type=['mp4', 'avi', 'mov', 'mkv'],
                key="video_upload"
            )
            if uploaded_file:
                video_source = "uploaded"
                
        elif source_type == "📷 Webcam":
            camera_id = st.number_input("Camera ID", min_value=0, max_value=10, value=0, key="camera_id")
            video_source = str(camera_id)
            
        else:  # URL/RTSP
            video_source = st.text_input(
                "Stream URL",
                placeholder="rtsp://... or http://...",
                key="video_url"
            )
        
        st.markdown("---")
        
        # Параметры детекции
        st.markdown("### 🎯 Detection Settings")
        
        # Выбор модели
        model_options = []
        if config.YOLO_TRAINED_MODEL.exists():
            model_options.append(str(config.YOLO_TRAINED_MODEL))
        if (config.MODELS_DIR / "discipline_monitor" / "weights" / "best.pt").exists():
            model_options.append(str(config.MODELS_DIR / "discipline_monitor" / "weights" / "best.pt"))
        
        # Добавляем предобученные модели YOLO
        for yolo_model in ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"]:
            if Path(yolo_model).exists():
                model_options.append(yolo_model)
        
        if model_options:
            selected_model = st.selectbox("YOLO Model", model_options, key="model_path")
        else:
            selected_model = "yolo11n.pt"
            st.warning("⚠️ No trained model found")
        
        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.1, max_value=1.0,
            value=config.YOLO_CONFIDENCE,
            step=0.05,
            key="confidence"
        )
        
        duration_threshold = st.slider(
            "Min Duration (sec)",
            min_value=0.5, max_value=10.0,
            value=float(config.VIOLATION_DURATION_THRESHOLD),
            step=0.5,
            key="duration_threshold"
        )
        
        st.markdown("---")
        
        # Настройки распознавания лиц
        st.markdown("### 👤 Face Recognition")
        
        if FACE_RECOGNITION_AVAILABLE:
            st.success(f"✅ Backend: {FACE_BACKEND}")
            
            enable_face = st.checkbox(
                "Enable Face Recognition",
                value=True,
                key="enable_face_recognition"
            )
            
            face_tolerance = st.slider(
                "Recognition Threshold",
                min_value=0.3, max_value=0.8,
                value=config.FACE_RECOGNITION_TOLERANCE,
                step=0.05,
                help="Lower = stricter",
                key="face_tolerance"
            )
            
            face_skip = st.slider(
                "Process every N frames",
                min_value=1, max_value=10,
                value=3,
                help="Higher = faster but less accurate",
                key="face_skip"
            )
        else:
            st.error("❌ Face recognition unavailable")
            enable_face = False
            face_tolerance = 0.5
            face_skip = 3
        
        st.markdown("---")
        
        # Информация о системе
        st.markdown("### 📊 System Info")
        
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            st.success(f"🎮 GPU: {gpu_name}")
        else:
            st.warning("⚠️ CPU mode")
        
        return {
            'source_type': source_type,
            'video_source': video_source,
            'uploaded_file': uploaded_file,
            'model_path': selected_model,
            'confidence': confidence,
            'duration_threshold': duration_threshold,
            'enable_face_recognition': enable_face,
            'face_tolerance': face_tolerance,
            'face_skip': face_skip
        }


# ==================== ГЛАВНЫЙ КОНТЕНТ ====================
def render_main_content(settings: dict):
    """Отрисовка главной страницы"""
    
    # Заголовок
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Discipline Monitoring System</h1>
        <p>Automatic violation detection with YOLO11 + Face Recognition</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Статистика
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status = "🟢 Active" if st.session_state.is_running else "🔴 Stopped"
        st.metric("Status", status)
    
    with col2:
        st.metric("Violations", st.session_state.total_violations)
    
    with col3:
        st.metric("FPS", f"{st.session_state.fps:.1f}")
    
    with col4:
        if st.session_state.start_time:
            elapsed = datetime.now() - st.session_state.start_time
            st.metric("Runtime", str(elapsed).split('.')[0])
        else:
            st.metric("Runtime", "00:00:00")
    
    st.markdown("---")
    
    # Основная область
    video_col, info_col = st.columns([2, 1])
    
    with video_col:
        st.markdown("### 📹 Video Stream")
        video_placeholder = st.empty()
        
        # Кнопки управления
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        
        with btn_col1:
            start_btn = st.button(
                "▶️ Start" if not st.session_state.is_running else "⏸️ Pause",
                type="primary",
                use_container_width=True
            )
        
        with btn_col2:
            stop_btn = st.button("⏹️ Stop", type="secondary", use_container_width=True)
        
        with btn_col3:
            clear_btn = st.button("🗑️ Clear Log", use_container_width=True)
    
    with info_col:
        # Вкладки: журнал и распознанные лица
        tab1, tab2 = st.tabs(["📋 Violations", "👤 Faces"])
        
        with tab1:
            if st.session_state.violations_log:
                for v in reversed(st.session_state.violations_log[-10:]):
                    st.markdown(f"""
                    <div class="violation-card">
                        <strong>{v['time']}</strong> - {v['type']}<br>
                        👤 {v['person']} | ⏱️ {v['duration']}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No violations detected yet")
        
        with tab2:
            if st.session_state.detected_faces:
                for face in st.session_state.detected_faces[-5:]:
                    name = face.get('name', 'Unknown')
                    conf = face.get('confidence', 0)
                    color = "#28a745" if name != "Unknown" else "#dc3545"
                    st.markdown(f"""
                    <div style="background: {color}22; border-left: 3px solid {color}; 
                                padding: 8px; margin: 5px 0; border-radius: 5px;">
                        👤 <strong>{name}</strong> ({conf:.0%})
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No faces detected yet")
        
        # Статистика по типам
        st.markdown("### 📊 Statistics")
        if st.session_state.violations_by_type:
            stats_df = pd.DataFrame([
                {"Type": k, "Count": v}
                for k, v in st.session_state.violations_by_type.items()
            ])
            st.bar_chart(stats_df.set_index("Type"))
    
    # Обработка кнопок
    if start_btn:
        st.session_state.is_running = not st.session_state.is_running
        if st.session_state.is_running:
            st.session_state.start_time = datetime.now()
        st.rerun()
    
    if stop_btn:
        st.session_state.is_running = False
        st.rerun()
    
    if clear_btn:
        st.session_state.violations_log = []
        st.session_state.total_violations = 0
        st.session_state.violations_by_type = defaultdict(int)
        st.session_state.violations_by_person = defaultdict(int)
        st.session_state.detected_faces = []
        st.rerun()
    
    return video_placeholder


def run_detection(video_placeholder, settings: dict):
    """Запуск детекции в реальном времени"""
    
    # Загрузка модели
    model = load_model(settings['model_path'])
    if model is None:
        st.error("❌ Failed to load model")
        return
    
    # Загрузка базы лиц
    face_db = None
    if settings['enable_face_recognition'] and FACE_RECOGNITION_AVAILABLE:
        face_db = load_face_database()
        if face_db:
            persons = face_db.list_persons()
            st.sidebar.info(f"👤 People in DB: {len(persons)}")
    
    # Открытие видеоисточника
    cap = None
    temp_file = None
    
    if settings['source_type'] == "📁 Video File" and settings['uploaded_file']:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_file.write(settings['uploaded_file'].read())
        temp_file.close()
        cap = cv2.VideoCapture(temp_file.name)
        
    elif settings['source_type'] == "📷 Webcam":
        cap = cv2.VideoCapture(int(settings['video_source'] or 0))
        
    elif settings['source_type'] == "🌐 URL/RTSP" and settings['video_source']:
        cap = cv2.VideoCapture(settings['video_source'])
    
    if cap is None or not cap.isOpened():
        st.error("❌ Failed to open video source")
        if temp_file:
            os.unlink(temp_file.name)
        return
    
    # Информация о видео
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    # Прогресс для видеофайлов
    if settings['source_type'] == "📁 Video File" and total_frames > 0:
        progress_bar = st.progress(0)
        progress_text = st.empty()
    else:
        progress_bar = None
        progress_text = None
    
    frame_count = 0
    start_time = time.time()
    face_frame_counter = 0
    
    try:
        while st.session_state.is_running and cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                if settings['source_type'] == "📁 Video File":
                    st.success("✅ Video processing complete!")
                    st.session_state.is_running = False
                break
            
            frame_count += 1
            face_frame_counter += 1
            annotated_frame = frame.copy()
            
            # === ДЕТЕКЦИЯ YOLO ===
            results = model.track(frame, conf=settings['confidence'], persist=True, verbose=False)
            
            detections = []
            violation_boxes = []
            
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    track_id = int(box.id[0]) if box.id is not None else -1
                    
                    class_name = config.VIOLATION_CLASSES.get(cls_id, "unknown")
                    label = config.VIOLATION_LABELS.get(class_name, class_name)
                    color = config.COLORS.get(class_name, (128, 128, 128))
                    
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'class': class_name,
                        'label': label,
                        'confidence': conf,
                        'track_id': track_id,
                        'color': color
                    })
                    
                    violation_boxes.append((track_id, x1, y1, x2, y2))
                    
                    # Отрисовка бокса нарушения
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Подпись с длительностью
                    duration_str = ""
                    if track_id in st.session_state.active_violations:
                        viol = st.session_state.active_violations[track_id]
                        dur = (datetime.now() - viol['start_time']).total_seconds()
                        duration_str = f" | {dur:.0f}s"
                    
                    text = f"{label} {conf:.2f}{duration_str}"
                    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated_frame, (x1, y1 - text_h - 10), (x1 + text_w + 5, y1), color, -1)
                    cv2.putText(annotated_frame, text, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # === РАСПОЗНАВАНИЕ ЛИЦ ===
            recognized_faces = []
            if settings['enable_face_recognition'] and face_db and face_frame_counter % settings['face_skip'] == 0:
                recognized_faces = detect_and_recognize_faces(
                    frame, face_db, settings['face_tolerance']
                )
                
                # Обновляем список распознанных лиц
                if recognized_faces:
                    st.session_state.detected_faces = recognized_faces
                
                # Отрисовка лиц
                annotated_frame = draw_face_annotations(annotated_frame, recognized_faces)
            
            # === ОБНОВЛЕНИЕ ЖУРНАЛА НАРУШЕНИЙ ===
            current_tracks = set()
            
            for det in detections:
                track_id = det['track_id']
                if track_id < 0:
                    continue
                    
                current_tracks.add(track_id)
                
                # Найти ближайшее лицо для этого нарушения
                person_name = "Unknown"
                if recognized_faces:
                    min_dist = float('inf')
                    vx1, vy1, vx2, vy2 = det['bbox']
                    vcx, vcy = (vx1 + vx2) // 2, (vy1 + vy2) // 2
                    
                    for face in recognized_faces:
                        fx1, fy1, fx2, fy2 = face['bbox']
                        fcx, fcy = (fx1 + fx2) // 2, (fy1 + fy2) // 2
                        dist = np.sqrt((vcx - fcx)**2 + (vcy - fcy)**2)
                        
                        if dist < min_dist and dist < 300:  # Макс. расстояние
                            min_dist = dist
                            person_name = face['name']
                
                # Новое нарушение
                if track_id not in st.session_state.active_violations:
                    st.session_state.active_violations[track_id] = {
                        'type': det['label'],
                        'start_time': datetime.now(),
                        'confidence': det['confidence'],
                        'person': person_name,
                        'logged': False
                    }
                else:
                    # Обновление существующего
                    violation = st.session_state.active_violations[track_id]
                    duration = (datetime.now() - violation['start_time']).total_seconds()
                    
                    # Обновляем имя если распознали
                    if person_name != "Unknown":
                        violation['person'] = person_name
                    
                    # Если превысили порог и ещё не залогировали
                    if duration >= settings['duration_threshold'] and not violation['logged']:
                        violation['logged'] = True
                        
                        log_entry = {
                            'id': len(st.session_state.violations_log) + 1,
                            'time': violation['start_time'].strftime("%H:%M:%S"),
                            'type': violation['type'],
                            'confidence': f"{violation['confidence']:.2f}",
                            'duration': f"{duration:.1f}s",
                            'person': violation['person']
                        }
                        
                        st.session_state.violations_log.append(log_entry)
                        st.session_state.total_violations += 1
                        st.session_state.violations_by_type[violation['type']] += 1
                        st.session_state.violations_by_person[violation['person']] += 1
            
            # Удаление завершённых нарушений
            finished = [tid for tid in st.session_state.active_violations if tid not in current_tracks]
            for tid in finished:
                del st.session_state.active_violations[tid]
            
            # === ОТОБРАЖЕНИЕ ===
            elapsed = time.time() - start_time
            if elapsed > 0:
                st.session_state.fps = frame_count / elapsed
            
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(annotated_frame_rgb, channels="RGB", use_container_width=True)
            
            # Прогресс
            if progress_bar and total_frames > 0:
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                progress_text.text(f"Progress: {frame_count}/{total_frames} ({progress*100:.1f}%)")
            
            st.session_state.processed_frames = frame_count
            time.sleep(0.001)
            
    except Exception as e:
        st.error(f"❌ Error: {e}")
        import traceback
        st.code(traceback.format_exc())
    
    finally:
        cap.release()
        if temp_file:
            try:
                os.unlink(temp_file.name)
            except:
                pass
        if progress_bar:
            progress_bar.empty()
        if progress_text:
            progress_text.empty()


# ==================== СТРАНИЦА ОТЧЁТОВ ====================
def render_reports_page():
    """Страница отчётов"""
    st.markdown("## 📊 Reports & Export")
    
    if not st.session_state.violations_log:
        st.info("No violations recorded. Start detection to collect data.")
        return
    
    # Статистика
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Violations", st.session_state.total_violations)
    with col2:
        if st.session_state.start_time:
            elapsed = datetime.now() - st.session_state.start_time
            st.metric("Monitoring Time", str(elapsed).split('.')[0])
    with col3:
        st.metric("Frames Processed", st.session_state.processed_frames)
    
    st.markdown("---")
    
    # Таблица
    st.markdown("### 📋 Full Log")
    df = pd.DataFrame(st.session_state.violations_log)
    st.dataframe(df, use_container_width=True)
    
    # Графики
    st.markdown("### 📈 Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### By Violation Type")
        if st.session_state.violations_by_type:
            chart_data = pd.DataFrame([
                {"Type": k, "Count": v}
                for k, v in st.session_state.violations_by_type.items()
            ])
            st.bar_chart(chart_data.set_index("Type"))
    
    with col2:
        st.markdown("#### By Person")
        if st.session_state.violations_by_person:
            chart_data = pd.DataFrame([
                {"Person": k, "Count": v}
                for k, v in st.session_state.violations_by_person.items()
            ])
            st.bar_chart(chart_data.set_index("Person"))
    
    st.markdown("---")
    
    # Экспорт
    st.markdown("### 💾 Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        json_data = json.dumps(st.session_state.violations_log, ensure_ascii=False, indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name=f"violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col2:
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col3:
        try:
            from io import BytesIO
            buffer = BytesIO()
            df.to_excel(buffer, index=False, engine='openpyxl')
            st.download_button(
                label="📥 Download Excel",
                data=buffer.getvalue(),
                file_name=f"violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except ImportError:
            st.info("Install openpyxl for Excel export")


# ==================== СТРАНИЦА БАЗЫ ЛИЦ ====================
def render_face_database_page():
    """Страница управления базой лиц"""
    st.markdown("## 👤 Face Database")
    
    if not FACE_RECOGNITION_AVAILABLE:
        st.error(f"❌ Face recognition unavailable. Backend: {FACE_BACKEND}")
        st.info("Install: pip install face-recognition dlib")
        return
    
    # Кнопка перезагрузки базы
    col_reload, col_info = st.columns([1, 3])
    with col_reload:
        if st.button("🔄 Reload Database"):
            load_face_database.clear()
            st.rerun()
    
    face_db = load_face_database()
    
    if face_db is None:
        st.error("❌ Failed to load face database")
        return
    
    with col_info:
        st.info(f"🔧 Backend: **{FACE_BACKEND}**")
    
    # Список людей
    persons = face_db.list_persons()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📋 People in Database")
        
        if persons:
            for person in persons:
                count = len(face_db.face_encodings.get(person, [])) if hasattr(face_db, 'face_encodings') else 1
                st.markdown(f"👤 **{person}** ({count} embeddings)")
        else:
            st.warning("Database is empty")
        
        st.markdown(f"**Total:** {len(persons)} people")
    
    with col2:
        st.markdown("### ➕ Add Person")
        
        name = st.text_input("Name", placeholder="John Doe", key="add_name")
        photo = st.file_uploader("Photo", type=['jpg', 'jpeg', 'png'], key="add_photo")
        
        if st.button("Add to Database", type="primary"):
            if name and photo:
                # Сохранение фото
                temp_path = Path(tempfile.gettempdir()) / f"face_{name}.jpg"
                with open(temp_path, 'wb') as f:
                    f.write(photo.read())
                
                try:
                    with st.spinner(f"Adding {name}..."):
                        if face_db.add_person(name, str(temp_path)):
                            face_db.save_database()
                            st.success(f"✅ {name} added successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Face not found in image")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                finally:
                    temp_path.unlink(missing_ok=True)
            else:
                st.warning("⚠️ Enter name and upload photo")
        
        st.markdown("---")
        
        # Удаление
        st.markdown("### 🗑️ Remove Person")
        
        if persons:
            person_to_delete = st.selectbox("Select person to remove", persons, key="delete_person")
            if st.button("Remove", type="secondary"):
                face_db.remove_person(person_to_delete)
                face_db.save_database()
                st.success(f"✅ {person_to_delete} removed")
                st.rerun()
    
    st.markdown("---")
    
    # Тест распознавания через камеру
    st.markdown("### 🧪 Test Recognition")
    
    if st.button("📷 Capture & Recognize"):
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Captured Image:**")
                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                with col2:
                    st.markdown("**Recognition Results:**")
                    
                    with st.spinner("Recognizing faces..."):
                        faces = detect_and_recognize_faces(frame, face_db, config.FACE_RECOGNITION_TOLERANCE)
                    
                    if faces:
                        for face in faces:
                            name = face.get('name', 'Unknown')
                            conf = face.get('confidence', 0)
                            
                            if name != "Unknown":
                                st.success(f"✅ **{name}** ({conf:.0%})")
                            else:
                                st.warning(f"❓ Unknown face (dist: {face.get('distance', 0):.2f})")
                    else:
                        st.info("No faces detected")
        else:
            st.error("❌ Failed to open camera")


# ==================== СТРАНИЦА НАСТРОЕК ====================
def render_settings_page():
    """Страница настроек"""
    st.markdown("## ⚙️ Settings")
    
    st.markdown("### 🎯 YOLO Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_input("Default Model", value=config.YOLO_MODEL, disabled=True)
        st.number_input("IOU Threshold", value=config.YOLO_IOU, disabled=True)
        st.number_input("Image Size", value=config.YOLO_IMG_SIZE, disabled=True)
    
    with col2:
        st.text_input("Trained Model Path", value=str(config.YOLO_TRAINED_MODEL), disabled=True)
        st.checkbox("Use GPU", value=config.USE_GPU, disabled=True)
    
    st.markdown("---")
    
    st.markdown("### 📹 Violation Classes")
    
    for cls_id, cls_name in config.VIOLATION_CLASSES.items():
        col1, col2, col3 = st.columns([1, 2, 2])
        with col1:
            st.text(f"ID: {cls_id}")
        with col2:
            st.text(f"Class: {cls_name}")
        with col3:
            st.text(f"Label: {config.VIOLATION_LABELS.get(cls_name, cls_name)}")
    
    st.markdown("---")
    
    st.markdown("### 📁 Paths")
    
    st.text_input("Data Directory", value=str(config.DATA_DIR), disabled=True)
    st.text_input("Models Directory", value=str(config.MODELS_DIR), disabled=True)
    st.text_input("Output Directory", value=str(config.OUTPUT_DIR), disabled=True)
    
    st.markdown("---")
    
    st.markdown("### 🔧 System Information")
    
    import torch
    import sys
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"🐍 Python: {sys.version.split()[0]}")
        st.info(f"🔥 PyTorch: {torch.__version__}")
        st.info(f"👤 Face Backend: {FACE_BACKEND}")
    
    with col2:
        if torch.cuda.is_available():
            st.success(f"🎮 CUDA: {torch.version.cuda}")
            st.success(f"💻 GPU: {torch.cuda.get_device_name(0)}")
        else:
            st.warning("⚠️ CUDA not available")


# ==================== ГЛАВНАЯ ФУНКЦИЯ ====================
def main():
    """Main application function"""
    
    init_session_state()
    apply_custom_styles()
    
    # Навигация
    page = st.sidebar.selectbox(
        "📍 Navigation",
        ["🏠 Home", "📊 Reports", "👤 Face Database", "⚙️ Settings"],
        key="navigation"
    )
    
    if page == "🏠 Home":
        settings = render_sidebar()
        video_placeholder = render_main_content(settings)
        
        if st.session_state.is_running:
            run_detection(video_placeholder, settings)
    
    elif page == "📊 Reports":
        render_reports_page()
    
    elif page == "👤 Face Database":
        render_face_database_page()
    
    elif page == "⚙️ Settings":
        render_settings_page()
    
    # Футер
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    <div style="text-align: center; color: #888; font-size: 0.8em;">
        🎯 Discipline Monitoring System<br>
        v2.0 | YOLO11 + {FACE_BACKEND}
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
