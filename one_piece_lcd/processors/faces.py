"""Face detection and CLIP embedding processor for anime characters."""

import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Import anime-face-detector (requires PYTHONOPTIMIZE=1 to bypass version checks)
from anime_face_detector import create_detector


class AnimeFaceProcessor:
    """Processor for detecting anime faces and generating CLIP embeddings."""
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the face processor.
        
        Args:
            device: Device to use ('cuda' or 'cpu'). Defaults to 'cpu' for compatibility.
        """
        # Force CPU for now - RTX 5090 needs PyTorch 2.7+ but anime-face-detector needs older mm* packages
        self.device = device or "cpu"
        print(f"AnimeFaceProcessor using device: {self.device}", file=sys.stderr)
        
        # Lazy-load models
        self._detector = None
        self._clip_model = None
        self._clip_processor = None
        
    @property
    def detector(self):
        """Lazy-load anime face detector."""
        if self._detector is None:
            print("Loading anime face detector...", file=sys.stderr)
            self._detector = create_detector("yolov3", device=self.device)
        return self._detector
    
    @property
    def clip_model(self):
        """Lazy-load CLIP model."""
        if self._clip_model is None:
            print("Loading CLIP model (downloading if needed)...", file=sys.stderr)
            self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self._clip_model = self._clip_model.to(self.device)
            self._clip_model.eval()
            print("CLIP model loaded!", file=sys.stderr)
        return self._clip_model
    
    @property
    def clip_processor(self):
        """Lazy-load CLIP processor."""
        if self._clip_processor is None:
            self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return self._clip_processor
    
    def detect_faces(self, image_path: str | Path) -> list[dict]:
        """
        Detect anime faces in an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of detected faces with bounding boxes and keypoints
        """
        image = cv2.imread(str(image_path))
        if image is None:
            return []
        
        preds = self.detector(image)
        return preds
    
    def crop_face(
        self,
        image_path: str | Path,
        bbox: np.ndarray,
        padding: float = 0.2
    ) -> Optional[Image.Image]:
        """
        Crop a face from an image given a bounding box.
        
        Args:
            image_path: Path to the image file
            bbox: Bounding box [x1, y1, x2, y2, confidence]
            padding: Padding ratio to add around the face
            
        Returns:
            Cropped face as PIL Image, or None if failed
        """
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        h, w = image.shape[:2]
        x1, y1, x2, y2 = map(int, bbox[:4])
        
        # Add padding
        face_w = x2 - x1
        face_h = y2 - y1
        pad_w = int(face_w * padding)
        pad_h = int(face_h * padding)
        
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)
        
        # Crop
        face = image[y1:y2, x1:x2]
        
        # Convert to PIL
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        return Image.fromarray(face_rgb)
    
    def crop_and_save_faces(
        self,
        image_path: str | Path,
        output_dir: Optional[Path] = None,
        min_confidence: float = 0.5
    ) -> list[Path]:
        """
        Detect faces in an image, crop them, and save to disk.
        
        Args:
            image_path: Path to source image
            output_dir: Directory to save cropped faces (defaults to same as source)
            min_confidence: Minimum detection confidence threshold
            
        Returns:
            List of paths to saved face images
        """
        image_path = Path(image_path)
        if output_dir is None:
            output_dir = image_path.parent
        
        preds = self.detect_faces(image_path)
        saved_paths = []
        
        for i, pred in enumerate(preds):
            bbox = pred["bbox"]
            confidence = bbox[4]
            
            if confidence < min_confidence:
                continue
            
            face_img = self.crop_face(image_path, bbox)
            if face_img is None:
                continue
            
            # Generate output filename
            stem = image_path.stem
            face_path = output_dir / f"{stem}_face_{i + 1}.png"
            face_img.save(face_path)
            saved_paths.append(face_path)
        
        return saved_paths
    
    def generate_clip_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Generate a CLIP embedding for an image.
        
        Args:
            image: PIL Image
            
        Returns:
            Normalized embedding as numpy array (512 dimensions)
        """
        inputs = self.clip_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            embedding = self.clip_model.get_image_features(**inputs)
        
        # Normalize the embedding
        embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten()
    
    def generate_embedding_from_path(self, image_path: str | Path) -> Optional[np.ndarray]:
        """
        Generate CLIP embedding from an image file path.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Embedding array, or None if failed
        """
        try:
            image = Image.open(image_path).convert("RGB")
            return self.generate_clip_embedding(image)
        except Exception as e:
            print(f"Error generating embedding for {image_path}: {e}", file=sys.stderr)
            return None
    
    def cache_embedding(self, image_path: str | Path, embedding: np.ndarray) -> Path:
        """
        Save embedding to disk as .npy file.
        
        Args:
            image_path: Original image path (used to derive cache path)
            embedding: Embedding array to save
            
        Returns:
            Path to saved .npy file
        """
        image_path = Path(image_path)
        cache_path = image_path.with_suffix(".npy")
        np.save(cache_path, embedding)
        return cache_path
    
    def load_cached_embedding(self, image_path: str | Path) -> Optional[np.ndarray]:
        """
        Load cached embedding from disk.
        
        Args:
            image_path: Original image path (used to derive cache path)
            
        Returns:
            Cached embedding, or None if not found
        """
        cache_path = Path(image_path).with_suffix(".npy")
        if cache_path.exists():
            return np.load(cache_path)
        return None
    
    def get_or_create_embedding(
        self,
        image_path: str | Path,
        force_refresh: bool = False
    ) -> Optional[np.ndarray]:
        """
        Get embedding from cache or create new one.
        
        Args:
            image_path: Path to image
            force_refresh: If True, regenerate even if cached
            
        Returns:
            Embedding array, or None if failed
        """
        if not force_refresh:
            cached = self.load_cached_embedding(image_path)
            if cached is not None:
                return cached
        
        embedding = self.generate_embedding_from_path(image_path)
        if embedding is not None:
            self.cache_embedding(image_path, embedding)
        
        return embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def find_best_match(
    query_embedding: np.ndarray,
    reference_embeddings: dict[str, list[np.ndarray]],
    threshold: float = 0.7
) -> Optional[tuple[str, float]]:
    """
    Find the best matching character for a query embedding.
    
    Args:
        query_embedding: The embedding to match
        reference_embeddings: Dict of character_id -> list of embeddings
        threshold: Minimum similarity threshold
        
    Returns:
        Tuple of (character_id, similarity) or None if no match above threshold
    """
    best_match = None
    best_similarity = threshold
    
    for character_id, embeddings in reference_embeddings.items():
        for ref_embedding in embeddings:
            similarity = cosine_similarity(query_embedding, ref_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = character_id
    
    if best_match:
        return (best_match, best_similarity)
    return None

