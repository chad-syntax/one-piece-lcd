"""Face detection and image embedding processor for anime characters.

Uses YOLOv8 AnimeFace for detection and SigLIP for embeddings.
"""

import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoProcessor, AutoModel
from ultralytics import YOLO  # type: ignore[attr-defined]


class AnimeFaceProcessor:
    """Processor for detecting anime faces and generating image embeddings.
    
    Uses YOLOv8 AnimeFace for face detection and SigLIP for embeddings.
    """
    
    # Model identifiers
    YOLO_MODEL_REPO = "Fuyucchi/yolov8_animeface"
    YOLO_MODEL_FILE = "yolov8x6_animeface.pt"
    SIGLIP_MODEL = "google/siglip-base-patch16-224"
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the face processor.
        
        Args:
            device: Device to use ('cuda' or 'cpu'). Auto-detects if not specified.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(f"AnimeFaceProcessor using device: {self.device}", file=sys.stderr)
        
        # Lazy-load models
        self._detector = None
        self._embedding_model = None
        self._embedding_processor = None
        
    @property
    def detector(self) -> YOLO:
        """Lazy-load YOLOv8 AnimeFace detector."""
        if self._detector is None:
            print(f"[Model] Downloading YOLOv8 AnimeFace from {self.YOLO_MODEL_REPO}...", file=sys.stderr)
            sys.stderr.flush()
            
            # Download model from HuggingFace
            model_path = hf_hub_download(
                repo_id=self.YOLO_MODEL_REPO,
                filename=self.YOLO_MODEL_FILE
            )
            print(f"[Model] Downloaded to: {model_path}", file=sys.stderr)
            
            print("[Model] Initializing YOLO detector...", file=sys.stderr)
            sys.stderr.flush()
            # YOLO will use device parameter during inference, no need for .to()
            self._detector = YOLO(model_path)
            print(f"[Model] YOLOv8 AnimeFace detector ready on {self.device}!", file=sys.stderr)
            sys.stderr.flush()
        return self._detector
    
    @property
    def embedding_model(self):
        """Lazy-load SigLIP model for embeddings."""
        if self._embedding_model is None:
            print(f"[Model] Downloading SigLIP model from {self.SIGLIP_MODEL}...", file=sys.stderr)
            sys.stderr.flush()
            self._embedding_model = AutoModel.from_pretrained(self.SIGLIP_MODEL)
            print(f"[Model] Moving SigLIP to {self.device}...", file=sys.stderr)
            sys.stderr.flush()
            self._embedding_model = self._embedding_model.to(self.device)
            self._embedding_model.eval()
            # Verify model is on correct device
            model_device = next(self._embedding_model.parameters()).device
            print(f"[Model] SigLIP model ready on {model_device}!", file=sys.stderr)
            sys.stderr.flush()
        return self._embedding_model
    
    @property
    def embedding_processor(self):
        """Lazy-load SigLIP processor."""
        if self._embedding_processor is None:
            print(f"[Model] Loading SigLIP processor from {self.SIGLIP_MODEL}...", file=sys.stderr)
            sys.stderr.flush()
            self._embedding_processor = AutoProcessor.from_pretrained(self.SIGLIP_MODEL, use_fast=True)
            print("[Model] SigLIP processor ready!", file=sys.stderr)
            sys.stderr.flush()
        return self._embedding_processor
    
    def detect_faces(
        self,
        image: np.ndarray | str | Path,
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        imgsz: int = 640,
        augment: bool = False,
        max_det: int = 100,
    ) -> list[dict]:
        """
        Detect anime faces in an image.
        
        Args:
            image: Image as numpy array (BGR), or path to image file
            conf_threshold: Minimum confidence threshold (lower = more detections)
            iou_threshold: IOU threshold for NMS (lower = allow more overlapping boxes)
            imgsz: Input image size for detection (larger = more accurate, slower)
            augment: Enable test-time augmentation (TTA) for better accuracy
            max_det: Maximum number of detections per image
            
        Returns:
            List of detected faces with bounding boxes:
            [{"bbox": [x1, y1, x2, y2, conf], ...}, ...]
        """
        if isinstance(image, (str, Path)):
            loaded = cv2.imread(str(image))  # type: ignore[attr-defined]
            if loaded is None:
                return []
            image = loaded
        
        # Run detection with tunable parameters
        # Pass device explicitly to ensure GPU usage (use 0 for cuda:0)
        device_param = 0 if self.device == "cuda" and torch.cuda.is_available() else self.device
        results = self.detector(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            augment=augment,
            max_det=max_det,
            device=device_param,  # YOLO expects 0 for cuda:0, or "cpu"
            verbose=False,
        )
        
        faces = []
        for result in results:
            if result.boxes is None:
                continue
            
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                
                faces.append({
                    "bbox": np.array([x1, y1, x2, y2, conf])
                })
        
        return faces
    
    def detect_faces_in_frame(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        imgsz: int = 640,
        augment: bool = False,
        max_det: int = 100,
    ) -> list[dict]:
        """
        Detect faces in a video frame (convenience method).
        
        Args:
            frame: BGR numpy array from cv2.VideoCapture
            conf_threshold: Minimum confidence threshold
            iou_threshold: IOU threshold for NMS
            imgsz: Input image size for detection
            augment: Enable test-time augmentation
            max_det: Maximum number of detections
            
        Returns:
            List of detected faces with bounding boxes
        """
        return self.detect_faces(
            frame,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            imgsz=imgsz,
            augment=augment,
            max_det=max_det,
        )
    
    def detect_faces_batch(
        self,
        images: list[np.ndarray],
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        imgsz: int = 640,
        augment: bool = False,
        max_det: int = 100,
    ) -> list[list[dict]]:
        """
        Detect faces in a batch of images (much faster on GPU).
        
        Args:
            images: List of BGR numpy arrays from cv2.VideoCapture
            conf_threshold: Minimum confidence threshold
            iou_threshold: IOU threshold for NMS
            imgsz: Input image size for detection
            augment: Enable test-time augmentation
            max_det: Maximum number of detections per image
            
        Returns:
            List of detection results, one per input image
        """
        if not images:
            return []
        
        # Run batch detection (YOLOv8 handles batching automatically)
        # Pass device explicitly to ensure GPU usage (use 0 for cuda:0)
        device_param = 0 if self.device == "cuda" and torch.cuda.is_available() else self.device
        results = self.detector(
            images,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            augment=augment,
            max_det=max_det,
            device=device_param,  # YOLO expects 0 for cuda:0, or "cpu"
            verbose=False,
        )
        
        # Convert results to our format
        batch_detections = []
        for result in results:
            faces = []
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    faces.append({
                        "bbox": np.array([x1, y1, x2, y2, conf])
                    })
            batch_detections.append(faces)
        
        return batch_detections
    
    def crop_face(
        self,
        image: np.ndarray | str | Path,
        bbox: np.ndarray,
        padding: float = 0.2
    ) -> Optional[Image.Image]:
        """
        Crop a face from an image given a bounding box.
        
        Args:
            image: Image as numpy array (BGR), or path to image file
            bbox: Bounding box [x1, y1, x2, y2, confidence]
            padding: Padding ratio to add around the face
            
        Returns:
            Cropped face as PIL Image, or None if failed
        """
        if isinstance(image, (str, Path)):
            loaded = cv2.imread(str(image))  # type: ignore[attr-defined]
            if loaded is None:
                return None
            img_array: np.ndarray = loaded
        else:
            img_array = image
        
        h, w = img_array.shape[:2]
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
        face = img_array[y1:y2, x1:x2]
        if face.size == 0:
            return None
        
        # Convert to PIL
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # type: ignore[attr-defined]
        return Image.fromarray(face_rgb)
    
    def crop_and_save_faces(
        self,
        image_path: str | Path,
        output_dir: Optional[Path] = None,
        min_confidence: float = 0.3
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
        
        # Load image once
        loaded = cv2.imread(str(image_path))  # type: ignore[attr-defined]
        if loaded is None:
            return []
        image: np.ndarray = loaded
        
        preds = self.detect_faces(image, conf_threshold=min_confidence)
        saved_paths = []
        
        for i, pred in enumerate(preds):
            bbox = pred["bbox"]
            
            face_img = self.crop_face(image, bbox)
            if face_img is None:
                continue
            
            # Generate output filename
            stem = image_path.stem
            face_path = output_dir / f"{stem}_face_{i + 1}.png"
            face_img.save(face_path)
            saved_paths.append(face_path)
        
        return saved_paths
    
    def generate_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Generate a SigLIP embedding for an image.
        
        Args:
            image: PIL Image
            
        Returns:
            Normalized embedding as numpy array
        """
        inputs = self.embedding_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.embedding_model.get_image_features(**inputs)
        
        # Normalize the embedding
        embedding = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten()
    
    def generate_embeddings_batch(self, images: list[Image.Image]) -> list[np.ndarray]:
        """
        Generate SigLIP embeddings for multiple images in a single batch.
        
        Args:
            images: List of PIL Images
            
        Returns:
            List of normalized embeddings as numpy arrays
        """
        if not images:
            return []
        
        inputs = self.embedding_processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.embedding_model.get_image_features(**inputs)
        
        # Normalize the embeddings (keep on GPU for now)
        embeddings = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        # Transfer to CPU in one batch operation
        embeddings_np = embeddings.cpu().numpy()
        
        return [embeddings_np[i].flatten() for i in range(len(images))]
    
    def generate_embedding_from_path(self, image_path: str | Path) -> Optional[np.ndarray]:
        """
        Generate SigLIP embedding from an image file path.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Embedding array, or None if failed
        """
        try:
            image = Image.open(image_path).convert("RGB")
            return self.generate_embedding(image)
        except Exception as e:
            print(f"Error generating embedding for {image_path}: {e}", file=sys.stderr)
            return None
    
    def cache_embedding(self, image_path: str | Path, embedding: np.ndarray, suffix: str = ".npy") -> Path:
        """
        Save embedding to disk as .npy file.
        
        Args:
            image_path: Original image path (used to derive cache path)
            embedding: Embedding array to save
            suffix: File suffix for cache file
            
        Returns:
            Path to saved .npy file
        """
        image_path = Path(image_path)
        cache_path = image_path.with_suffix(suffix)
        np.save(cache_path, embedding)
        return cache_path
    
    def load_cached_embedding(self, image_path: str | Path, suffix: str = ".npy") -> Optional[np.ndarray]:
        """
        Load cached embedding from disk.
        
        Args:
            image_path: Original image path (used to derive cache path)
            suffix: File suffix for cache file
            
        Returns:
            Cached embedding, or None if not found
        """
        cache_path = Path(image_path).with_suffix(suffix)
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


from typing import Sequence, Mapping

def find_all_matches(
    query_embedding: np.ndarray,
    reference_embeddings: Mapping[str, Sequence[np.ndarray | torch.Tensor]],
    threshold: float = 0.5,
    device: Optional[str] = None,
) -> list[tuple[str, float]]:
    """
    Find all matching characters above threshold for a query embedding.
    
    Args:
        query_embedding: The embedding to match
        reference_embeddings: Dict of character_id -> list of embeddings
        threshold: Minimum similarity threshold
        device: Device to use for GPU acceleration ('cuda' or None for CPU)
        
    Returns:
        List of (character_id, best_similarity) tuples, sorted by similarity descending
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Use GPU-accelerated matching if available
    if device == "cuda" and torch.cuda.is_available():
        # Defensive: convert sequences/possible tensors to ndarrays if needed
        ref_dict = {k: [e.detach().cpu().numpy() if isinstance(e, torch.Tensor) else e for e in v] for k, v in reference_embeddings.items()}
        return _find_all_matches_gpu(query_embedding, ref_dict, threshold, device)
    else:
        ref_dict = {k: [e.detach().cpu().numpy() if isinstance(e, torch.Tensor) else e for e in v] for k, v in reference_embeddings.items()}
        return _find_all_matches_cpu(query_embedding, ref_dict, threshold)


def _find_all_matches_cpu(
    query_embedding: np.ndarray,
    reference_embeddings: dict[str, list[np.ndarray]],
    threshold: float = 0.5
) -> list[tuple[str, float]]:
    """CPU-based matching (original implementation)."""
    matches: dict[str, float] = {}
    
    for character_id, embeddings in reference_embeddings.items():
        best_sim = 0.0
        for ref_embedding in embeddings:
            similarity = cosine_similarity(query_embedding, ref_embedding)
            if similarity > best_sim:
                best_sim = similarity
        
        if best_sim >= threshold:
            matches[character_id] = best_sim
    
    # Sort by similarity descending
    sorted_matches = sorted(matches.items(), key=lambda x: x[1], reverse=True)
    return sorted_matches


def _find_all_matches_gpu(
    query_embedding: np.ndarray,
    reference_embeddings: dict[str, list[np.ndarray]],
    threshold: float = 0.5,
    device: str = "cuda",
) -> list[tuple[str, float]]:
    """GPU-accelerated matching using PyTorch."""
    # Convert query to tensor
    query_tensor = torch.from_numpy(query_embedding).float().to(device)
    
    matches: dict[str, float] = {}
    
    # Process all embeddings for each character at once
    for character_id, embeddings in reference_embeddings.items():
        if not embeddings:
            continue
        
        # Stack all reference embeddings for this character
        ref_tensor = torch.from_numpy(np.stack(embeddings)).float().to(device)
        
        # Compute cosine similarity (dot product since embeddings are normalized)
        similarities = torch.matmul(ref_tensor, query_tensor)
        
        # Get best similarity
        best_sim = float(torch.max(similarities).cpu().item())
        
        if best_sim >= threshold:
            matches[character_id] = best_sim
    
    # Sort by similarity descending
    sorted_matches = sorted(matches.items(), key=lambda x: x[1], reverse=True)
    return sorted_matches


from typing import Sequence, Any

from typing import Sequence, Mapping

def find_all_matches_batch(
    query_embeddings: Sequence[np.ndarray | torch.Tensor],
    reference_embeddings: Mapping[str, Sequence[np.ndarray | torch.Tensor]],
    threshold: float = 0.5,
    device: Optional[str] = None,
) -> list[list[tuple[str, float]]]:
    """
    Find matches for a batch of query embeddings (much faster on GPU).
    
    Args:
        query_embeddings: List of embeddings to match
        reference_embeddings: Dict of character_id -> list of embeddings
        threshold: Minimum similarity threshold
        device: Device to use for GPU acceleration
    
    Returns:
        List of match results, one per query embedding
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not query_embeddings:
        return []
    
    if device == "cuda" and torch.cuda.is_available():
        # If both queries and refs are torch tensors, skip conversion
        # Assume pre-loaded as CUDA tensors
        use_cuda_tensors = isinstance(next(iter(reference_embeddings.values()))[0], torch.Tensor)
        return _find_all_matches_batch_gpu(query_embeddings, reference_embeddings, threshold, device, use_cuda_tensors)
    else:
        # Fallback to CPU for each
        # Fix type: Only pass the np.ndarray branch to CPU matcher
        results = []
        for emb in query_embeddings:
            if isinstance(emb, torch.Tensor):
                emb_np = emb.detach().cpu().numpy()
            else:
                emb_np = emb
            # Convert Mapping[str, Sequence[ndarray|Tensor]] -> dict[str, list[ndarray]]
            ref_np = {k: [e.detach().cpu().numpy() if isinstance(e, torch.Tensor) else e for e in v] for k, v in reference_embeddings.items()}
            results.append(_find_all_matches_cpu(emb_np, ref_np, threshold))
        return results



from typing import Mapping

def _find_all_matches_batch_gpu(
    query_embeddings: Sequence[np.ndarray | torch.Tensor],
    reference_embeddings: Mapping[str, Sequence[np.ndarray | torch.Tensor]],
    threshold: float = 0.5,
    device: str = "cuda",
    use_cuda_tensors: bool = False,
) -> list[list[tuple[str, float]]]:
    """GPU-accelerated batch matching (now supports persistent CUDA tensors)."""
    # Defensive: ensure all query embeddings are tensors or ndarrays
    if use_cuda_tensors:
        # Assume already loaded as CUDA tensors (avoid conversion/copy)
        query_tensor = torch.stack([e if isinstance(e, torch.Tensor) else torch.from_numpy(e) for e in query_embeddings]).float().to(device)
    else:
        query_tensor = torch.from_numpy(np.stack([e if isinstance(e, np.ndarray) else e.detach().cpu().numpy() for e in query_embeddings])).float().to(device)

    all_ref_embeddings: list[torch.Tensor] = []
    char_ids: list[str] = []
    for character_id, embeddings in reference_embeddings.items():
        if not embeddings:
            continue
        if use_cuda_tensors:
            ref_tensor = torch.stack([e if isinstance(e, torch.Tensor) else torch.from_numpy(e) for e in embeddings]).float().to(device)
        else:
            ref_tensor = torch.from_numpy(np.stack([e if isinstance(e, np.ndarray) else e.detach().cpu().numpy() for e in embeddings])).float().to(device)
        all_ref_embeddings.append(ref_tensor)
        char_ids.append(character_id)
    
    if not all_ref_embeddings:
        return [[] for _ in query_embeddings]
    
    all_best_sims = []
    for ref_tensor in all_ref_embeddings:
        similarities = torch.matmul(query_tensor, ref_tensor.t())
        best_sims = torch.max(similarities, dim=1)[0]
        all_best_sims.append(best_sims)
    if all_best_sims:
        all_best_sims_tensor = torch.stack(all_best_sims)  # (num_chars, batch_size)
        all_best_sims_cpu = all_best_sims_tensor.cpu().numpy()  # Single CPU transfer
    else:
        all_best_sims_cpu = np.array([])
    results = []
    for i in range(len(query_embeddings)):
        matches = []
        for char_idx, char_id in enumerate(char_ids):
            sim = float(all_best_sims_cpu[char_idx, i])
            if sim >= threshold:
                matches.append((char_id, sim))
        matches.sort(key=lambda x: x[1], reverse=True)
        results.append(matches)
    return results
