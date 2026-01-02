"""Video character recognition using YOLOv8 AnimeFace + SigLIP embeddings."""

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image

from ..constants.paths import SAMPLE_VIDEO_DIR, FACE_DETECTIONS_FILENAME_PATTERN
from ..models.video import (
    CharacterCandidate,
    CharacterMatch,
    Coordinates,
    FaceDetection,
    FrameData,
    VideoFaceRecognitionResult,
    VideoProcessingConfig,
)
from ..utils.characters import (
    get_character_face_embedding_paths,
    get_character_image_embedding_paths,
)
from .faces import AnimeFaceProcessor, cosine_similarity, find_all_matches


class VideoFaceRecognition:
    """Character recognition for video files using YOLOv8 + SigLIP."""
    
    def __init__(
        self,
        video_path: str | Path,
        character_ids: List[str],
        affiliations: List[str],
        frame_skip: int = 5,
        face_tolerance: float = 0.5,
        character_tolerance: float = 0.5,
        face_detection_threshold: float = 0.3,
    ):
        """
        Initialize the character recognition system.
        
        Args:
            video_path: Path to the video file
            character_ids: List of character IDs to match against
            affiliations: List of affiliation IDs being searched
            frame_skip: Process every Nth frame (default 5)
            face_tolerance: Similarity threshold for face embedding matches (default 0.5)
            character_tolerance: Similarity threshold for full-image matches (default 0.5)
            face_detection_threshold: Minimum confidence for face detection (default 0.3)
        """
        self.video_path = str(video_path)
        self.character_ids = character_ids
        self.affiliations = affiliations
        self.frame_skip = frame_skip
        self.face_tolerance = face_tolerance
        self.character_tolerance = character_tolerance
        self.face_detection_threshold = face_detection_threshold
        
        # Embeddings: char_id -> list of embeddings
        self.face_embeddings: dict[str, list[np.ndarray]] = {}
        self.full_image_embeddings: dict[str, list[np.ndarray]] = {}
        
        self.frame_data: List[FrameData] = []
        self.detected_characters: set[str] = set()
        self.total_frames_processed: int = 0
        self.video_duration_seconds: float = 0.0
        self.processing_duration_seconds: float = 0.0
        self.processor: AnimeFaceProcessor = AnimeFaceProcessor()
        self.output_path: Path | None = None
        
        # Load embeddings on init
        self._load_embeddings()
        
    def _load_embeddings(self) -> None:
        """Load cached embeddings for face images and full character images."""
        print("Loading cached SigLIP embeddings...", file=sys.stderr)
        sys.stderr.flush()
        
        face_count = 0
        full_count = 0
        chars_with_embeddings = 0
        
        for character_id in self.character_ids:
            char_face_embeddings = []
            char_full_embeddings = []
            
            # Load face embeddings directly from embedding paths in character.json
            face_embedding_paths = get_character_face_embedding_paths(character_id)
            for npy_path in face_embedding_paths:
                try:
                    embedding = np.load(npy_path)
                    char_face_embeddings.append(embedding)
                    face_count += 1
                except Exception as e:
                    print(f"  ✗ Error loading {npy_path.name}: {e}", file=sys.stderr)
            
            # Load full image embeddings directly from embedding paths in character.json
            image_embedding_paths = get_character_image_embedding_paths(character_id)
            for npy_path in image_embedding_paths:
                try:
                    embedding = np.load(npy_path)
                    char_full_embeddings.append(embedding)
                    full_count += 1
                except Exception as e:
                    print(f"  ✗ Error loading {npy_path.name}: {e}", file=sys.stderr)
            
            if char_face_embeddings:
                self.face_embeddings[character_id] = char_face_embeddings
            if char_full_embeddings:
                self.full_image_embeddings[character_id] = char_full_embeddings
            
            if char_face_embeddings or char_full_embeddings:
                chars_with_embeddings += 1
        
        print(f"Loaded {face_count} face + {full_count} full-image embeddings for {chars_with_embeddings} characters\n", file=sys.stderr)
        
    def process_video(self) -> None:
        """Process the video: detect faces and match against embeddings."""
        start_time = time.time()
        
        print(f"Processing video: {self.video_path}", file=sys.stderr)
        
        video = cv2.VideoCapture(self.video_path)  # type: ignore[attr-defined]
        fps = video.get(cv2.CAP_PROP_FPS)  # type: ignore[attr-defined]
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # type: ignore[attr-defined]
        self.video_duration_seconds = total_frames / fps if fps > 0 else 0
        
        print(f"Video info: {fps:.1f} fps, {total_frames} frames", file=sys.stderr)
        print(f"Processing every {self.frame_skip} frames...\n", file=sys.stderr)
        
        if not self.face_embeddings and not self.full_image_embeddings:
            print("No embeddings loaded, cannot process video.", file=sys.stderr)
            return
        
        frame_count = 0
        processed_count = 0
        
        while True:
            success, frame = video.read()
            if not success:
                break
            
            if frame_count % self.frame_skip == 0:
                frame_result = self._process_frame(frame)
                
                # Only add frames with detections
                if frame_result.faces or frame_result.characters:
                    frame_result.frame_number = frame_count
                    self.frame_data.append(frame_result)
                
                processed_count += 1
                if processed_count % 20 == 0:
                    progress = (frame_count / total_frames) * 100
                    elapsed = time.time() - start_time
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) - {elapsed:.1f}s", file=sys.stderr)
            
            frame_count += 1
        
        video.release()
        self.total_frames_processed = processed_count
        self.processing_duration_seconds = time.time() - start_time
        print(f"\n✓ Video processing complete in {self.processing_duration_seconds:.1f}s!\n", file=sys.stderr)
    
    def _process_frame(self, frame: np.ndarray) -> FrameData:
        """Process a single frame: detect faces + match full frame."""
        faces: List[FaceDetection] = []
        characters: List[CharacterMatch] = []
        
        # 1. Detect faces and match against face embeddings
        if self.face_embeddings:
            preds = self.processor.detect_faces_in_frame(frame, self.face_detection_threshold)
            
            face_crops: list[Image.Image] = []
            face_coords: list[tuple[int, int, int, int]] = []
            
            for pred in preds:
                bbox = pred["bbox"]
                face_img = self.processor.crop_face(frame, bbox)
                if face_img is not None:
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    face_crops.append(face_img)
                    face_coords.append((x1, y1, x2, y2))
            
            if face_crops:
                face_embs = self.processor.generate_embeddings_batch(face_crops)
                
                for emb, (x1, y1, x2, y2) in zip(face_embs, face_coords):
                    matches = find_all_matches(emb, self.face_embeddings, self.face_tolerance)
                    
                    candidates: List[CharacterCandidate] = []
                    for char_id, similarity in matches:
                        candidates.append(CharacterCandidate(
                            character_id=char_id,
                            confidence=round(similarity, 3),
                        ))
                        self.detected_characters.add(char_id)
                    
                    # If no matches, mark as unknown with best similarity
                    if not candidates and self.face_embeddings:
                        best_matches = find_all_matches(emb, self.face_embeddings, 0.0)
                        if best_matches:
                            candidates.append(CharacterCandidate(
                                character_id="unknown",
                                confidence=round(best_matches[0][1], 3),
                            ))
                    
                    faces.append(FaceDetection(
                        coordinates=Coordinates(
                            top=y1, right=x2, bottom=y2, left=x1,
                            width=x2 - x1, height=y2 - y1,
                        ),
                        candidates=candidates,
                    ))
        
        # 2. Match full frame against full-image embeddings
        if self.full_image_embeddings:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # type: ignore[attr-defined]
            frame_pil = Image.fromarray(frame_rgb)
            
            frame_emb = self.processor.generate_embedding(frame_pil)
            matches = find_all_matches(frame_emb, self.full_image_embeddings, self.character_tolerance)
            
            for char_id, similarity in matches:
                characters.append(CharacterMatch(
                    character_id=char_id,
                    confidence=round(similarity, 3),
                    match_type="full_image",
                ))
                self.detected_characters.add(char_id)
        
        return FrameData(frame_number=0, faces=faces, characters=characters)
    
    def get_results(self) -> VideoFaceRecognitionResult:
        """Get the recognition results as a Pydantic model."""
        config = VideoProcessingConfig(
            frame_skip=self.frame_skip,
            processing_duration_seconds=round(self.processing_duration_seconds, 2),
            face_detection_threshold=self.face_detection_threshold,
            face_match_tolerance=self.face_tolerance,
            character_match_tolerance=self.character_tolerance,
            target_affiliations=self.affiliations,
            target_characters=self.character_ids,
            detected_characters=sorted(self.detected_characters),
        )
        
        return VideoFaceRecognitionResult(
            video_path=self.video_path,
            total_frames=self.total_frames_processed,
            config=config,
            frame_data=self.frame_data,
        )
    
    def save_results(self) -> Path:
        """Save results to JSON file and return the output path."""
        SAMPLE_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = FACE_DETECTIONS_FILENAME_PATTERN.format(timestamp=timestamp)
        self.output_path = SAMPLE_VIDEO_DIR / filename
        
        results = self.get_results()
        with open(self.output_path, "w") as f:
            f.write(results.model_dump_json(indent=2))
        
        return self.output_path
    
    def print_summary(self) -> None:
        """Print a one-line summary of results to stderr."""
        duration_minutes = self.video_duration_seconds / 60
        output_str = f" Saved report at {self.output_path}" if self.output_path else ""
        
        print(
            f"Processed {self.total_frames_processed} frames "
            f"({duration_minutes:.1f} min video) in {self.processing_duration_seconds:.1f}s, "
            f"recognized {len(self.detected_characters)} characters.{output_str}",
            file=sys.stderr
        )
