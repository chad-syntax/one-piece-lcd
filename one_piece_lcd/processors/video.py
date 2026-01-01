"""Video facial recognition processor using anime-face-detector and CLIP embeddings."""

import sys
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np

from ..utils.characters import get_character_face_paths
from .faces import AnimeFaceProcessor, cosine_similarity


class VideoFaceRecognition:
    """Facial recognition system for video files using CLIP embeddings."""
    
    def __init__(self, video_path: str | Path):
        """
        Initialize the facial recognition system.
        
        Args:
            video_path: Path to the video file
        """
        self.video_path = str(video_path)
        self.known_embeddings: dict[str, list[np.ndarray]] = {}  # character_id -> list of CLIP embeddings
        self.frame_detections: list[dict] = []
        self.processor: AnimeFaceProcessor | None = None
        
    def load_known_faces(self, character_ids: list[str]) -> None:
        """
        Load cached CLIP embeddings for the specified characters.
        
        Args:
            character_ids: List of character IDs to load embeddings for
        """
        print("Loading cached CLIP embeddings...", file=sys.stderr)
        
        total_loaded = 0
        total_embeddings = 0
        
        for character_id in character_ids:
            face_paths = get_character_face_paths(character_id)
            
            if not face_paths:
                continue
            
            character_embeddings = []
            
            for face_path in face_paths:
                # Look for cached .npy embedding
                npy_path = face_path.with_suffix(".npy")
                if npy_path.exists():
                    try:
                        embedding = np.load(npy_path)
                        character_embeddings.append(embedding)
                    except Exception as e:
                        print(f"  ✗ Error loading {npy_path.name}: {e}", file=sys.stderr)
            
            if character_embeddings:
                self.known_embeddings[character_id] = character_embeddings
                total_loaded += 1
                total_embeddings += len(character_embeddings)
        
        print(f"Loaded {total_embeddings} embeddings for {total_loaded} characters\n", file=sys.stderr)
        
    def process_video(self, frame_skip: int = 5, tolerance: float = 0.6) -> None:
        """
        Process the video and detect faces frame by frame.
        
        Args:
            frame_skip: Process every Nth frame (default 5)
            tolerance: CLIP similarity threshold (0-1, higher = stricter, default 0.6)
        """
        from PIL import Image
        
        print(f"Processing video: {self.video_path}", file=sys.stderr)
        
        # Initialize face processor if needed
        if self.processor is None:
            self.processor = AnimeFaceProcessor()
        
        video = cv2.VideoCapture(self.video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {fps:.1f} fps, {total_frames} frames", file=sys.stderr)
        print(f"Processing every {frame_skip} frames...\n", file=sys.stderr)
        
        # Flatten all known embeddings for comparison
        all_embeddings = []
        embedding_to_character = []
        
        for character_id, embeddings in self.known_embeddings.items():
            for embedding in embeddings:
                all_embeddings.append(embedding)
                embedding_to_character.append(character_id)
        
        if not all_embeddings:
            print("No embeddings loaded, cannot process video.", file=sys.stderr)
            return
        
        frame_count = 0
        processed_count = 0
        
        while True:
            success, frame = video.read()
            
            if not success:
                break
            
            if frame_count % frame_skip == 0:
                timestamp = frame_count / fps
                
                # Detect anime faces in frame
                preds = self.processor.detector(frame)
                
                frame_data = {
                    "timestamp": round(timestamp, 2),
                    "people": []
                }
                
                for pred in preds:
                    bbox = pred["bbox"]
                    x1, y1, x2, y2, conf = bbox[:5]
                    
                    if conf < 0.5:  # Skip low confidence detections
                        continue
                    
                    # Crop face from frame
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    h, w = frame.shape[:2]
                    
                    # Add padding
                    pad = int((x2 - x1) * 0.2)
                    x1 = max(0, x1 - pad)
                    y1 = max(0, y1 - pad)
                    x2 = min(w, x2 + pad)
                    y2 = min(h, y2 + pad)
                    
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        continue
                    
                    # Convert to PIL and generate CLIP embedding
                    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    face_pil = Image.fromarray(face_rgb)
                    face_embedding = self.processor.generate_clip_embedding(face_pil)
                    
                    # Find best match among known embeddings
                    best_similarity = 0.0
                    best_character = None
                    
                    for i, known_embedding in enumerate(all_embeddings):
                        similarity = cosine_similarity(face_embedding, known_embedding)
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_character = embedding_to_character[i]
                    
                    # Check if it's a valid match (above threshold)
                    if best_similarity >= tolerance and best_character:
                        frame_data["people"].append({
                            "character_id": best_character,
                            "confidence": round(best_similarity, 3),
                            "coordinates": {
                                "top": y1,
                                "right": x2,
                                "bottom": y2,
                                "left": x1,
                                "width": x2 - x1,
                                "height": y2 - y1
                            }
                        })
                
                self.frame_detections.append(frame_data)
                
                processed_count += 1
                if processed_count % 20 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)", file=sys.stderr)
            
            frame_count += 1
        
        video.release()
        print("\n✓ Video processing complete!\n", file=sys.stderr)
        
    def create_time_segments(self, gap_threshold: float = 1.0) -> list[dict]:
        """
        Convert frame detections into time segments with all people in each segment.
        
        Args:
            gap_threshold: Max gap (seconds) to consider frames part of same segment
        
        Returns:
            List of segments with structure:
            {
                start_time: float,
                end_time: float,
                duration: float,
                matches: [{character_id: str, coordinates: {...}, confidence: float}]
            }
        """
        if not self.frame_detections:
            return []
        
        segments: list[dict] = []
        current_segment: dict | None = None
        segment_people: dict[str, list[dict]] = {}
        
        for frame in self.frame_detections:
            timestamp = frame["timestamp"]
            people_in_frame = set(p["character_id"] for p in frame["people"])
            
            # Check if we should start a new segment
            should_start_new = False
            
            if current_segment is None:
                should_start_new = True
            else:
                # Check time gap
                time_gap = timestamp - current_segment["end_time"]
                if time_gap > gap_threshold:
                    should_start_new = True
                else:
                    # Check if the set of people changed
                    current_people = set(segment_people.keys())
                    if people_in_frame != current_people:
                        should_start_new = True
            
            if should_start_new:
                # Save previous segment if it exists
                if current_segment:
                    segments.append(self._finalize_segment(current_segment, segment_people))
                
                # Start new segment
                current_segment = {
                    "start_time": timestamp,
                    "end_time": timestamp,
                    "frame_count": 1
                }
                segment_people = {p["character_id"]: [p] for p in frame["people"]}
            else:
                # Extend current segment
                current_segment["end_time"] = timestamp
                current_segment["frame_count"] += 1
                
                # Accumulate people data for averaging
                for person in frame["people"]:
                    if person["character_id"] in segment_people:
                        segment_people[person["character_id"]].append(person)
        
        # Don't forget the last segment
        if current_segment:
            segments.append(self._finalize_segment(current_segment, segment_people))
        
        return segments
    
    def _finalize_segment(self, segment: dict, segment_people: dict[str, list[dict]]) -> dict:
        """Average coordinates and confidence for all people in segment."""
        matches = []
        
        for character_id, person_frames in segment_people.items():
            # Average coordinates across all frames
            avg_coords = {
                "top": 0, "right": 0, "bottom": 0, "left": 0,
                "width": 0, "height": 0
            }
            avg_confidence = 0.0
            
            for frame_data in person_frames:
                for key in avg_coords.keys():
                    avg_coords[key] += frame_data["coordinates"][key]
                avg_confidence += frame_data["confidence"]
            
            count = len(person_frames)
            avg_coords = {k: int(v / count) for k, v in avg_coords.items()}
            avg_confidence = round(avg_confidence / count, 3)
            
            matches.append({
                "character_id": character_id,
                "coordinates": avg_coords,
                "confidence": avg_confidence
            })
        
        return {
            "start_time": segment["start_time"],
            "end_time": segment["end_time"],
            "duration": round(segment["end_time"] - segment["start_time"], 2),
            "matches": matches
        }
        
    def get_results(self, gap_threshold: float = 1.0) -> dict:
        """
        Get the recognition results as a JSON-serializable dict.
        
        Args:
            gap_threshold: Max gap (seconds) for segment grouping
            
        Returns:
            Dict with video path and time segments
        """
        segments = self.create_time_segments(gap_threshold)
        
        return {
            "video": self.video_path,
            "total_segments": len(segments),
            "segments": segments
        }
    
    def print_summary(self, gap_threshold: float = 1.0) -> None:
        """Print a human-readable summary of results to stderr."""
        segments = self.create_time_segments(gap_threshold)
        
        print("=" * 70, file=sys.stderr)
        print("TIME SEGMENTS SUMMARY", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        
        for i, seg in enumerate(segments):
            start_fmt = str(timedelta(seconds=int(seg["start_time"])))
            end_fmt = str(timedelta(seconds=int(seg["end_time"])))
            
            people_names = [m["character_id"] for m in seg["matches"]]
            
            print(f"\nSegment {i + 1}: {start_fmt} → {end_fmt} ({seg['duration']}s)", file=sys.stderr)
            print(f"  People: {', '.join(people_names)}", file=sys.stderr)
            
            for match in seg["matches"]:
                print(f"    • {match['character_id']}", file=sys.stderr)
                print(f"      Confidence: {match['confidence']}", file=sys.stderr)

