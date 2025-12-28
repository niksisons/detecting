"""
Face database management using DeepFace (Facenet512)
"""
import os
import pickle
from pathlib import Path
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import config

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from deepface import DeepFace


class FaceDatabase:
    """Class for face database operations using DeepFace"""
    
    def __init__(self, db_path: Path = None, model_name: str = "Facenet512"):
        """
        Initialize face database
        
        Args:
            db_path: Path to database folder
            model_name: DeepFace model name (Facenet512, VGG-Face, ArcFace, etc.)
        """
        self.db_path = db_path or config.FACES_DB_DIR
        self.encodings_file = self.db_path / "face_encodings_deepface.pkl"
        self.model_name = model_name
        self.face_encodings: Dict[str, List[np.ndarray]] = {}
        self.detector_backend = "mtcnn"  # mtcnn, retinaface, opencv
        
        # Load database
        self.load_database()
        
        # Pre-load model (first call is slow)
        print(f"Loading DeepFace model: {model_name}...")
        try:
            # Build model on startup
            DeepFace.build_model(model_name)
            print(f"Model {model_name} loaded successfully")
        except Exception as e:
            print(f"Warning: Could not preload model: {e}")
    
    def extract_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extract face embedding from image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Face embedding vector or None if face not found
        """
        try:
            # Get embedding using DeepFace
            result = DeepFace.represent(
                img_path=image_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=True,
                align=True
            )
            
            if result and len(result) > 0:
                return np.array(result[0]["embedding"])
            return None
            
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None
    
    def extract_embedding_from_frame(self, frame: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Extract face embeddings from video frame
        
        Args:
            frame: BGR image (OpenCV format)
            
        Returns:
            List of tuples (embedding, bbox) where bbox is (x, y, w, h)
        """
        results = []
        
        try:
            # Get embeddings and face regions
            embeddings = DeepFace.represent(
                img_path=frame,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=True
            )
            
            for emb_data in embeddings:
                embedding = np.array(emb_data["embedding"])
                facial_area = emb_data.get("facial_area", {})
                
                # Get bounding box
                x = facial_area.get("x", 0)
                y = facial_area.get("y", 0)
                w = facial_area.get("w", 0)
                h = facial_area.get("h", 0)
                
                if w > 0 and h > 0:
                    results.append((embedding, (x, y, w, h)))
            
        except Exception as e:
            # No faces detected or other error
            pass
            
        return results
    
    def add_person(self, name: str, image_path: str) -> bool:
        """
        Add person to database from single photo
        
        Args:
            name: Person name
            image_path: Path to photo
            
        Returns:
            True if successful
        """
        print(f"Adding {name}...")
        
        embedding = self.extract_embedding(image_path)
        
        if embedding is not None:
            self.face_encodings.setdefault(name, []).append(embedding)
            print(f"  {name} added successfully (embedding: {len(embedding)} dims)")
            return True
        else:
            print(f"  Face not found in {image_path}")
            return False
    
    def add_person_from_folder(self, name: str, folder_path: str) -> int:
        """
        Add multiple photos of one person from folder
        
        Args:
            name: Person name
            folder_path: Path to folder with photos
            
        Returns:
            Number of successfully added images
        """
        folder = Path(folder_path)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        added_count = 0
        for img_file in folder.iterdir():
            if img_file.suffix.lower() in image_extensions:
                if self.add_person(name, str(img_file)):
                    added_count += 1
        
        print(f"Added {added_count} images for {name}")
        return added_count
    
    def recognize_face(self, face_embedding: np.ndarray, threshold: float = None) -> Tuple[str, float]:
        """
        Recognize face by embedding
        
        Args:
            face_embedding: Face embedding vector
            threshold: Recognition threshold (lower = stricter)
            
        Returns:
            Tuple[str, float]: (Name or "Unknown", distance)
        """
        if threshold is None:
            threshold = config.FACE_RECOGNITION_TOLERANCE
        
        if not self.face_encodings:
            return "Unknown", 1.0
        
        # Normalize input embedding
        face_embedding = face_embedding / np.linalg.norm(face_embedding)
        
        min_distance = float('inf')
        recognized_name = "Unknown"
        
        for name, encodings_list in self.face_encodings.items():
            for known_embedding in encodings_list:
                # Normalize known embedding
                known_norm = known_embedding / np.linalg.norm(known_embedding)
                
                # Cosine distance (1 - cosine similarity)
                cosine_sim = np.dot(face_embedding, known_norm)
                distance = 1 - cosine_sim
                
                if distance < min_distance:
                    min_distance = distance
                    if distance < threshold:
                        recognized_name = name
        
        return recognized_name, min_distance
    
    def save_database(self):
        """Save database to disk"""
        with open(self.encodings_file, 'wb') as f:
            pickle.dump({
                'encodings': self.face_encodings,
                'model_name': self.model_name
            }, f)
        print(f"Database saved: {self.encodings_file}")
    
    def load_database(self):
        """Load database from disk"""
        if self.encodings_file.exists():
            try:
                with open(self.encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, dict) and 'encodings' in data:
                        self.face_encodings = data['encodings']
                        saved_model = data.get('model_name', 'unknown')
                        if saved_model != self.model_name:
                            print(f"Warning: Database was created with {saved_model}, using {self.model_name}")
                    else:
                        # Old format compatibility
                        self.face_encodings = data
                print(f"Database loaded: {len(self.face_encodings)} persons")
            except Exception as e:
                print(f"Error loading database: {e}")
                self.face_encodings = {}
        else:
            print("Database empty, created new")
            self.face_encodings = {}
    
    def list_persons(self) -> List[str]:
        """Get list of all people in database"""
        return list(self.face_encodings.keys())
    
    def recognize_faces_in_frame(self, frame: np.ndarray, threshold: float = None) -> List[dict]:
        """
        Detect and recognize all faces in video frame
        
        Args:
            frame: BGR image (OpenCV format)
            threshold: Recognition threshold (lower = stricter)
            
        Returns:
            List of dicts with keys: name, distance, confidence, bbox, embedding
        """
        if threshold is None:
            threshold = config.FACE_RECOGNITION_TOLERANCE
        
        results = []
        
        # Extract all faces and their embeddings
        face_data = self.extract_embedding_from_frame(frame)
        
        for embedding, (x, y, w, h) in face_data:
            # Recognize this face
            name, distance = self.recognize_face(embedding, threshold)
            
            results.append({
                "name": name,
                "distance": distance,
                "confidence": max(0, 1 - distance),
                "bbox": (x, y, x + w, y + h),  # Convert to (x1, y1, x2, y2)
                "embedding": embedding
            })
        
        return results
    
    def remove_person(self, name: str) -> bool:
        """Remove person from database"""
        if name in self.face_encodings:
            del self.face_encodings[name]
            print(f"{name} removed from database")
            return True
        return False
    
    def merge_persons(self, names_to_merge: List[str], new_name: str):
        """
        Merge multiple persons into one
        
        Args:
            names_to_merge: List of person names to merge
            new_name: New combined name
        """
        combined_embeddings = []
        
        for name in names_to_merge:
            if name in self.face_encodings:
                combined_embeddings.extend(self.face_encodings[name])
                del self.face_encodings[name]
        
        if combined_embeddings:
            self.face_encodings[new_name] = combined_embeddings
            print(f"Merged {len(names_to_merge)} persons into '{new_name}' ({len(combined_embeddings)} embeddings)")
            return True
        return False


def main():
    """Face database demo with DeepFace"""
    print("=" * 60)
    print("FACE DATABASE MANAGEMENT (DeepFace)")
    print("=" * 60)
    
    db = FaceDatabase()
    
    print("\nAvailable commands:")
    print("1. add - Add person (single photo)")
    print("2. add_folder - Add person (folder with photos)")
    print("3. list - Show all people in database")
    print("4. remove - Remove person")
    print("5. merge - Merge multiple persons into one")
    print("6. rebuild - Rebuild database from images folder")
    print("7. exit - Exit")
    
    while True:
        command = input("\nEnter command: ").strip().lower()
        
        if command == "add":
            name = input("Person name: ").strip()
            image_path = input("Photo path: ").strip()
            if db.add_person(name, image_path):
                db.save_database()
        
        elif command == "add_folder":
            name = input("Person name: ").strip()
            folder_path = input("Folder path: ").strip()
            if db.add_person_from_folder(name, folder_path):
                db.save_database()
        
        elif command == "list":
            persons = db.list_persons()
            print(f"\nTotal in database: {len(persons)} persons")
            for i, person in enumerate(persons, 1):
                count = len(db.face_encodings[person])
                print(f"{i}. {person} ({count} embeddings)")
        
        elif command == "remove":
            name = input("Name to remove: ").strip()
            if db.remove_person(name):
                db.save_database()
        
        elif command == "merge":
            print("Enter names to merge (comma-separated):")
            names_input = input().strip()
            names = [n.strip() for n in names_input.split(',')]
            new_name = input("New combined name: ").strip()
            if db.merge_persons(names, new_name):
                db.save_database()
        
        elif command == "rebuild":
            # Rebuild from images in faces_database folder
            print("Rebuilding database from images...")
            db.face_encodings = {}
            images_dir = db.db_path
            for img_file in images_dir.glob("*.jpg"):
                # Use filename without extension as name
                name = img_file.stem.split('_')[0]  # First part before underscore
                db.add_person(name, str(img_file))
            db.save_database()
            
        elif command == "exit":
            break
        
        else:
            print("Unknown command")


if __name__ == "__main__":
    main()
