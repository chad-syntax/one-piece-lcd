"""Pydantic models for video face recognition output."""

from typing import List

from pydantic import BaseModel, Field


class Coordinates(BaseModel):
    """Bounding box coordinates for a detected face."""
    
    top: int = Field(description="Top edge Y coordinate")
    right: int = Field(description="Right edge X coordinate")
    bottom: int = Field(description="Bottom edge Y coordinate")
    left: int = Field(description="Left edge X coordinate")
    width: int = Field(description="Width of bounding box")
    height: int = Field(description="Height of bounding box")


class CharacterCandidate(BaseModel):
    """A possible character match for a detected face or frame."""
    
    character_id: str = Field(description="ID of the matched character ('unknown' if no match)")
    confidence: float = Field(description="Match confidence score (0-1)")


class FaceDetection(BaseModel):
    """A detected face with possible character matches."""
    
    coordinates: Coordinates = Field(description="Bounding box of the detected face")
    candidates: List[CharacterCandidate] = Field(
        default_factory=list,
        description="Possible character matches from face embeddings, sorted by confidence"
    )
    
    @property
    def best_match(self) -> CharacterCandidate | None:
        """Get the highest confidence match."""
        return self.candidates[0] if self.candidates else None


class CharacterMatch(BaseModel):
    """A character match from full-frame or full-body embedding comparison."""
    
    character_id: str = Field(description="ID of the matched character")
    confidence: float = Field(description="Match confidence score (0-1)")
    match_type: str = Field(
        default="full_image",
        description="Type of embedding match: 'full_image' or 'face_image'"
    )


class FrameData(BaseModel):
    """Detection data for a single video frame."""
    
    frame_number: int = Field(description="Frame number in the video")
    faces: List[FaceDetection] = Field(
        default_factory=list,
        description="Detected faces with character candidates (from face detection + face embeddings)"
    )
    characters: List[CharacterMatch] = Field(
        default_factory=list,
        description="Characters matched from full-frame embedding comparison (no bounding box)"
    )


class VideoProcessingConfig(BaseModel):
    """Configuration used for video processing."""
    
    frame_skip: int = Field(description="Interval between processed frames")
    processing_duration_seconds: float = Field(description="How long the processing took in seconds")
    face_detection_threshold: float = Field(
        default=0.3,
        description="Confidence threshold for face detection"
    )
    face_match_tolerance: float = Field(
        default=0.5,
        description="Similarity threshold for face embedding matching"
    )
    character_match_tolerance: float = Field(
        default=0.5,
        description="Similarity threshold for full-frame character matching"
    )
    target_affiliations: List[str] = Field(
        default_factory=list,
        description="Affiliation IDs that were searched for"
    )
    target_characters: List[str] = Field(
        default_factory=list,
        description="Character IDs that were searched for"
    )
    detected_characters: List[str] = Field(
        default_factory=list,
        description="Unique character IDs that were actually detected"
    )


class VideoFaceRecognitionResult(BaseModel):
    """Complete result of video face recognition processing."""
    
    video_path: str = Field(description="Path to the processed video file")
    total_frames: int = Field(description="Total number of frames processed")
    config: VideoProcessingConfig = Field(description="Processing configuration and detection summary")
    frame_data: List[FrameData] = Field(
        default_factory=list,
        description="Per-frame detection data (only frames with detections)"
    )
