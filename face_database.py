"""
Face database management
"""
import os
import pickle
from pathlib import Path
import face_recognition
import cv2
import numpy as np
from typing import Dict, List, Tuple
import config


class FaceDatabase:
    """Class for face database operations"""
    
    def __init__(self, db_path: Path = None):
        self.db_path = db_path or config.FACES_DB_DIR
        self.encodings_file = self.db_path / "face_encodings.pkl"
        self.face_encodings: Dict[str, List] = {}
        self.load_database()
    
    def add_person(self, name: str, image_path: str):
        print(f"Adding {name}...")
        
        try:
            # Use built-in face_recognition function for loading
            rgb_image = face_recognition.load_image_file(image_path)
            
            # Resize if too large (>2000px)
            height, width = rgb_image.shape[:2]
            if max(height, width) > 2000:
                scale = 2000 / max(height, width)
                new_size = (int(width * scale), int(height * scale))
                rgb_image = cv2.resize(rgb_image, new_size)
                print(f"   Resized: {width}x{height} -> {new_size[0]}x{new_size[1]}")
            
            # Find faces
            face_locations = face_recognition.face_locations(rgb_image, model="hog")
            
            if not face_locations:
                print(f"Face not found in {image_path}")
                return False
            
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            if face_encodings:
                self.face_encodings.setdefault(name, []).append(face_encodings[0])
                print(f"{name} added successfully ({len(face_locations)} faces found)")
                return True
            else:
                print(f"Could not encode face")
                return False

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def add_person_from_folder(self, name: str, folder_path: str):
        """
        Add multiple photos of one person from folder
        
        Args:
            name: Person name
            folder_path: Path to folder with photos
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
    
    def recognize_face(self, face_encoding: np.ndarray, threshold: float = None) -> Tuple[str, float]:
        """
        Recognize face by encoding
        
        Args:
            face_encoding: Face encoding
            threshold: Recognition threshold (uses config default if None)
            
        Returns:
            Tuple[str, float]: (Name or "Unknown", distance)
        """
        if not self.face_encodings:
            return "Unknown", 1.0
        
        tolerance = threshold if threshold is not None else config.FACE_RECOGNITION_TOLERANCE
        
        # Compare with known faces
        min_distance = float('inf')
        recognized_name = "Unknown"
        
        for name, encodings_list in self.face_encodings.items():
            # Compare with all encodings of this person
            distances = face_recognition.face_distance(encodings_list, face_encoding)
            min_dist = min(distances)
            
            if min_dist < min_distance:
                min_distance = min_dist
                if min_dist < tolerance:
                    recognized_name = name
        
        return recognized_name, min_distance
    
    def recognize_faces_in_frame(self, frame: np.ndarray, threshold: float = None) -> List[Dict]:
        """
        Detect and recognize all faces in frame
        
        Args:
            frame: BGR image (OpenCV format)
            threshold: Recognition threshold
            
        Returns:
            List of dicts with name, distance, confidence, bbox
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize for faster processing
        scale = 1.0
        height, width = rgb_frame.shape[:2]
        if max(height, width) > 800:
            scale = 800 / max(height, width)
            small_frame = cv2.resize(rgb_frame, (0, 0), fx=scale, fy=scale)
        else:
            small_frame = rgb_frame
        
        # Detect faces
        face_locations = face_recognition.face_locations(small_frame, model="hog")
        
        if not face_locations:
            return []
        
        # Get encodings
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        
        results = []
        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            # Scale back to original size
            top = int(top / scale)
            right = int(right / scale)
            bottom = int(bottom / scale)
            left = int(left / scale)
            
            # Recognize
            name, distance = self.recognize_face(encoding, threshold)
            confidence = max(0, 1 - distance)
            
            results.append({
                "name": name,
                "distance": distance,
                "confidence": confidence,
                "bbox": (left, top, right, bottom),
                "encoding": encoding
            })
        
        return results
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in frame (without recognition)
        
        Returns:
            List of bboxes (left, top, right, bottom)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize for speed
        scale = 1.0
        height, width = rgb_frame.shape[:2]
        if max(height, width) > 800:
            scale = 800 / max(height, width)
            small_frame = cv2.resize(rgb_frame, (0, 0), fx=scale, fy=scale)
        else:
            small_frame = rgb_frame
        
        face_locations = face_recognition.face_locations(small_frame, model="hog")
        
        # Scale back
        return [(int(left/scale), int(top/scale), int(right/scale), int(bottom/scale)) 
                for top, right, bottom, left in face_locations]
    
    def save_database(self):
        """Save database to disk"""
        with open(self.encodings_file, 'wb') as f:
            pickle.dump(self.face_encodings, f)
        print(f"Database saved: {self.encodings_file}")
    
    def load_database(self):
        """Load database from disk"""
        if self.encodings_file.exists():
            with open(self.encodings_file, 'rb') as f:
                self.face_encodings = pickle.load(f)
            print(f"Database loaded: {len(self.face_encodings)} persons")
        else:
            print("Database empty, created new")
            self.face_encodings = {}
    
    def list_persons(self) -> List[str]:
        """Get list of all people in database"""
        return list(self.face_encodings.keys())
    
    def remove_person(self, name: str):
        """Remove person from database"""
        if name in self.face_encodings:
            del self.face_encodings[name]
            # Remove photo folder
            person_dir = self.db_path / name
            if person_dir.exists():
                import shutil
                shutil.rmtree(person_dir)
            print(f"{name} removed from database")
            return True
        return False


def main():
    """Face database demo"""
    print("=" * 60)
    print("FACE DATABASE MANAGEMENT")
    print("=" * 60)
    
    db = FaceDatabase()
    
    print("\nAvailable commands:")
    print("1. add - Add person (single photo)")
    print("2. add_folder - Add person (folder with photos)")
    print("3. list - Show all people in database")
    print("4. remove - Remove person")
    print("5. exit - Exit")
    
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
                print(f"{i}. {person} ({count} photos)")
        
        elif command == "remove":
            name = input("Name to remove: ").strip()
            if db.remove_person(name):
                db.save_database()
        
        elif command == "exit":
            break
        
        else:
            print("Unknown command")


if __name__ == "__main__":
    main()