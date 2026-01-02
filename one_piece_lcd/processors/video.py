"""
Video character recognition using YOLOv8 AnimeFace + SigLIP embeddings (CPU & GPU, fully batched/pipelined).
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import List, Optional, Sequence, Mapping
import cv2
import numpy as np
import torch
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
from .faces import AnimeFaceProcessor, find_all_matches, find_all_matches_batch

@dataclass
class GPUWorkItem:
    batch_id: int
    frame_batch: List[tuple[int, np.ndarray]]
    face_crops: List[Image.Image]
    face_coords: List[tuple[int, int, int, int]]
    frame_face_indices: List[List[int]]
    frame_pils: List[Image.Image]
    detections: List[List[dict]]

@dataclass
class GPUResult:
    batch_id: int
    frame_batch: List[tuple[int, np.ndarray]]
    detections: List[List[dict]]
    frame_face_indices: List[List[int]]
    all_face_coords: List[tuple[int, int, int, int]]
    batch_matches: List[List[tuple[str, float]]]
    batch_frame_matches: List[List[tuple[str, float]]]
    all_face_embs: Optional[List[np.ndarray]]


class VideoFaceRecognition:
    """Character recognition for video files using YOLOv8 + SigLIP, supporting CPU and GPU."""
    def __init__(
        self,
        video_path: str | Path,
        character_ids: List[str],
        affiliations: List[str],
        frame_skip: int = 5,
        face_tolerance: float = 0.5,
        character_tolerance: float = 0.5,
        face_detection_threshold: float = 0.3,
        face_iou_threshold: float = 0.5,
        face_imgsz: int = 640,
        face_augment: bool = False,
        batch_size: int = 8,
    ):
        self.video_path = str(video_path)
        self.character_ids = character_ids
        self.affiliations = affiliations
        self.frame_skip = frame_skip
        self.face_tolerance = face_tolerance
        self.character_tolerance = character_tolerance
        self.face_detection_threshold = face_detection_threshold
        self.face_iou_threshold = face_iou_threshold
        self.face_imgsz = face_imgsz
        self.face_augment = face_augment
        self.batch_size = batch_size
        # Embeddings
        from typing import Sequence

        self.face_embeddings: dict[str, Sequence[np.ndarray | torch.Tensor]] = {}
        self.full_image_embeddings: dict[str, Sequence[np.ndarray | torch.Tensor]] = {}
        self.frame_data: List[FrameData] = []
        self.detected_characters: set[str] = set()
        self.total_frames_processed: int = 0
        self.video_duration_seconds: float = 0.0
        self.processing_duration_seconds: float = 0.0
        self.processor: AnimeFaceProcessor = AnimeFaceProcessor()
        self.output_path: Optional[Path] = None
        # Timing/stat lists
        self.detect_timings = []
        self.frame_emb_timings = []
        self.frame_match_timings = []
        self.crop_timings = []
        self.face_emb_timings = []
        self.face_match_timings = []
        self.organize_timings = []
        self.batch_total_timings = []
        self._load_embeddings()

    def _load_embeddings(self) -> None:
        print("Loading cached SigLIP embeddings...", file=sys.stderr)
        sys.stderr.flush()
        face_count = 0
        full_count = 0
        chars_with_embeddings = 0
        for character_id in self.character_ids:
            char_face_embeddings: List[np.ndarray | torch.Tensor] = []
            char_full_embeddings: List[np.ndarray | torch.Tensor] = []
            for npy_path in get_character_face_embedding_paths(character_id):
                try:
                    embedding = np.load(npy_path)
                    if torch.cuda.is_available() and self.processor.device == "cuda":
                        embedding = torch.from_numpy(embedding).float().to(self.processor.device)
                    char_face_embeddings.append(embedding)
                    face_count += 1
                except Exception as e:
                    print(f"  ✗ Error loading {npy_path.name}: {e}", file=sys.stderr)
            for npy_path in get_character_image_embedding_paths(character_id):
                try:
                    embedding = np.load(npy_path)
                    if torch.cuda.is_available() and self.processor.device == "cuda":
                        embedding = torch.from_numpy(embedding).float().to(self.processor.device)
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
        start_time = time.time()
        print(f"Processing video: {self.video_path}", file=sys.stderr)
        video = cv2.VideoCapture(self.video_path)  # type: ignore[attr-defined]
        fps = video.get(cv2.CAP_PROP_FPS)  # type: ignore[attr-defined]
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # type: ignore[attr-defined]
        self.video_duration_seconds = total_frames / fps if fps > 0 else 0
        print(f"Video info: {fps:.1f} fps, {total_frames} frames", file=sys.stderr)
        print(f"Processing every {self.frame_skip} frames with batch size {self.batch_size}...", file=sys.stderr)
        if self.batch_size > 32:
            print(f"Warning: Large batch size ({self.batch_size}) may cause high memory usage.", file=sys.stderr)
        print("", file=sys.stderr)
        if not self.face_embeddings and not self.full_image_embeddings:
            print("No embeddings loaded, cannot process video.", file=sys.stderr)
            return
        # Queues for producer-consumer pattern
        frame_queue: Queue[Optional[tuple[int, np.ndarray]]] = Queue(maxsize=min(self.batch_size * 2, 32))
        gpu_work_queue: Queue[Optional[GPUWorkItem]] = Queue(maxsize=2)
        result_queue: Queue[Optional[GPUResult]] = Queue(maxsize=2)
        batch_counter_lock = Lock()
        batch_counter = [0]
        # Worker functions
        def read_frames():
            try:
                frame_count = 0
                while True:
                    success, frame = video.read()
                    if not success:
                        frame_queue.put(None)
                        break
                    if frame_count % self.frame_skip == 0:
                        try:
                            frame_queue.put((frame_count, frame.copy()), timeout=5.0)
                        except:
                            print(f"Warning: Frame queue full, dropping frame {frame_count}", file=sys.stderr)
                            break
                    frame_count += 1
            except Exception as e:
                print(f"Error in frame reader: {e}", file=sys.stderr)
            finally:
                video.release()
        def detection_worker():
            try:
                frame_batch: List[tuple[int, np.ndarray]] = []
                while True:
                    try:
                        item = frame_queue.get(timeout=1.0)
                    except Empty:
                        if frame_batch:
                            with batch_counter_lock:
                                batch_id = batch_counter[0]
                                batch_counter[0] += 1
                            work_item = self._prepare_gpu_work_item(frame_batch, batch_id)
                            if work_item:
                                try:
                                    gpu_work_queue.put(work_item, timeout=10.0)
                                except:
                                    print("Error: GPU work queue full, aborting", file=sys.stderr)
                                    break
                            frame_batch = []
                        continue
                    if item is None:
                        if frame_batch:
                            with batch_counter_lock:
                                batch_id = batch_counter[0]
                                batch_counter[0] += 1
                            work_item = self._prepare_gpu_work_item(frame_batch, batch_id)
                            if work_item:
                                try:
                                    gpu_work_queue.put(work_item, timeout=10.0)
                                except:
                                    print("Error: GPU work queue full, aborting", file=sys.stderr)
                        gpu_work_queue.put(None)
                        break
                    frame_batch.append(item)
                    if len(frame_batch) >= self.batch_size:
                        with batch_counter_lock:
                            batch_id = batch_counter[0]
                            batch_counter[0] += 1
                        work_item = self._prepare_gpu_work_item(frame_batch, batch_id)
                        if work_item:
                            try:
                                gpu_work_queue.put(work_item, timeout=10.0)
                            except:
                                print("Error: GPU work queue full, aborting", file=sys.stderr)
                                break
                        frame_batch = []
            except Exception as e:
                print(f"Error in detection worker: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
        def gpu_processor():
            try:
                while True:
                    work_item = gpu_work_queue.get()
                    if work_item is None:
                        result_queue.put(None)
                        break
                    try:
                        result = self._process_gpu_work_item(work_item)
                        result_queue.put(result, timeout=10.0)
                        del work_item
                        import gc
                        gc.collect()
                    except Exception as e:
                        print(f"Error processing GPU work item: {e}", file=sys.stderr)
                        import traceback
                        traceback.print_exc()
                    finally:
                        gpu_work_queue.task_done()
            except Exception as e:
                print(f"Error in GPU processor: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
        def result_writer():
            try:
                processed_count = 0
                while True:
                    result = result_queue.get()
                    if result is None:
                        break
                    try:
                        self._write_gpu_result(result)
                        processed_count += len(result.frame_batch)
                        if processed_count % 20 == 0:
                            progress = (processed_count * self.frame_skip / total_frames) * 100
                            elapsed = time.time() - start_time
                            print(f"Progress: {progress:.1f}% ({processed_count * self.frame_skip}/{total_frames}) - {elapsed:.1f}s", file=sys.stderr)
                        del result
                        import gc
                        gc.collect()
                    except Exception as e:
                        print(f"Error writing result: {e}", file=sys.stderr)
                    finally:
                        result_queue.task_done()
                self.total_frames_processed = processed_count
            except Exception as e:
                print(f"Error in result writer: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
        # Launch threads
        reader_thread = Thread(target=read_frames, daemon=False)
        detection_thread = Thread(target=detection_worker, daemon=False)
        gpu_thread = Thread(target=gpu_processor, daemon=False)
        writer_thread = Thread(target=result_writer, daemon=False)
        reader_thread.start()
        detection_thread.start()
        gpu_thread.start()
        writer_thread.start()
        reader_thread.join()
        detection_thread.join()
        gpu_thread.join()
        writer_thread.join()
        self.processing_duration_seconds = time.time() - start_time
        print(f"\n✓ Video processing complete in {self.processing_duration_seconds:.1f}s!\n", file=sys.stderr)
        self._print_timing_summary()

    def _prepare_gpu_work_item(self, frame_batch: List[tuple[int, np.ndarray]], batch_id: int) -> Optional[GPUWorkItem]:
        if not frame_batch:
            return None
        # --- Detection ---
        detect_start = time.time()
        frames_only = [frame for _, frame in frame_batch]
        batch_detections = self.processor.detect_faces_batch(
            frames_only,
            conf_threshold=self.face_detection_threshold,
            iou_threshold=self.face_iou_threshold,
            imgsz=self.face_imgsz,
            augment=self.face_augment,
        )
        detect_elapsed = time.time() - detect_start
        self.detect_timings.append(detect_elapsed)

        # --- Frame PIL Conversion (PreFrame Embedding) ---
        frame_emb_start = time.time()
        def convert_frame(frame_data):
            _, frame = frame_data
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb)
        frame_pils = list(ThreadPoolExecutor(max_workers=min(len(frame_batch), 4)).map(convert_frame, frame_batch))
        frame_emb_elapsed = time.time() - frame_emb_start
        self.frame_emb_timings.append(frame_emb_elapsed)

        # --- Cropping ---
        crop_start = time.time()
        all_face_crops, all_face_coords, frame_face_indices = [], [], []
        crop_tasks = []
        for frame_idx, ((frame_num, frame), detections) in enumerate(zip(frame_batch, batch_detections)):
            frame_face_indices.append([])
            for pred_idx, pred in enumerate(detections):
                bbox = pred["bbox"]
                crop_tasks.append((frame_idx, pred_idx, frame, bbox))
        def crop_single_face(task):
            frame_idx, pred_idx, frame, bbox = task
            face_img = self.processor.crop_face(frame, bbox)
            if face_img is not None:
                x1, y1, x2, y2 = map(int, bbox[:4])
                return (frame_idx, pred_idx, face_img, (x1, y1, x2, y2))
            return (frame_idx, pred_idx, None, None)
        if crop_tasks:
            crop_results = list(ThreadPoolExecutor(max_workers=min(len(crop_tasks), 16)).map(crop_single_face, crop_tasks))
            for frame_idx, pred_idx, face_img, coords in crop_results:
                if face_img is not None and coords is not None:
                    face_idx = len(all_face_crops)
                    all_face_crops.append(face_img)
                    all_face_coords.append(coords)
                    frame_face_indices[frame_idx].append(face_idx)
        crop_elapsed = time.time() - crop_start
        self.crop_timings.append(crop_elapsed)
        return GPUWorkItem(
            batch_id=batch_id,
            frame_batch=frame_batch,
            face_crops=all_face_crops,
            face_coords=all_face_coords,
            frame_face_indices=frame_face_indices,
            frame_pils=frame_pils,
            detections=batch_detections,
        )

    def _process_gpu_work_item(self, work_item: GPUWorkItem) -> GPUResult:
        batch_total_start = time.time()
        # Face embedding/match
        face_emb_start = time.time()
        all_face_embs = None
        batch_matches: List[List[tuple[str, float]]] = []
        if work_item.face_crops and self.face_embeddings:
            all_face_embs = self.processor.generate_embeddings_batch(work_item.face_crops)
            face_emb_elapsed = time.time() - face_emb_start
            self.face_emb_timings.append(face_emb_elapsed)
            face_match_start = time.time()
            batch_matches = find_all_matches_batch(
                all_face_embs,
                self.face_embeddings,
                self.face_tolerance,
                device=self.processor.device,
            )
            face_match_elapsed = time.time() - face_match_start
            self.face_match_timings.append(face_match_elapsed)
        else:
            batch_matches = [[] for _ in work_item.face_crops]
            self.face_emb_timings.append(0.0)
            self.face_match_timings.append(0.0)
        # Frame embedding/match
        frame_emb_start = time.time()
        batch_frame_matches: List[List[tuple[str, float]]] = []
        if self.full_image_embeddings and work_item.frame_pils:
            batch_frame_embs = self.processor.generate_embeddings_batch(work_item.frame_pils)
            frame_emb_elapsed = time.time() - frame_emb_start
            self.frame_emb_timings.append(frame_emb_elapsed)
            frame_match_start = time.time()
            batch_frame_matches = find_all_matches_batch(
                batch_frame_embs,
                self.full_image_embeddings,
                self.character_tolerance,
                device=self.processor.device,
            )
            frame_match_elapsed = time.time() - frame_match_start
            self.frame_match_timings.append(frame_match_elapsed)
        else:
            batch_frame_matches = [[] for _ in work_item.frame_pils]
            self.frame_emb_timings.append(0.0)
            self.frame_match_timings.append(0.0)
        # For organizing, just append 0 (not implemented)
        self.organize_timings.append(0.0)
        batch_total_elapsed = time.time() - batch_total_start
        self.batch_total_timings.append(batch_total_elapsed)
        return GPUResult(
            batch_id=work_item.batch_id,
            frame_batch=work_item.frame_batch,
            detections=work_item.detections,
            frame_face_indices=work_item.frame_face_indices,
            all_face_coords=work_item.face_coords,
            batch_matches=batch_matches,
            batch_frame_matches=batch_frame_matches,
            all_face_embs=all_face_embs,
        )


    def _write_gpu_result(self, result: GPUResult) -> None:
        for frame_idx, ((frame_num, frame), detections) in enumerate(zip(result.frame_batch, result.detections)):
            frame_result = self._process_frame_with_batched_matches(
                frame,
                detections,
                result.frame_face_indices[frame_idx],
                result.all_face_coords,
                result.batch_matches,
                result.all_face_embs,
                result.batch_frame_matches[frame_idx] if result.batch_frame_matches else None,
            )
            if frame_result.faces or frame_result.characters:
                frame_result.frame_number = frame_num
                self.frame_data.append(frame_result)

    def _process_frame_with_batched_matches(
        self,
        frame: np.ndarray,
        preds: List[dict],
        face_indices: List[int],
        all_face_coords: List[tuple[int, int, int, int]],
        batch_matches: List[List[tuple[str, float]]],
        all_face_embs: Optional[List[np.ndarray]] = None,
        frame_matches: Optional[List[tuple[str, float]]] = None,
    ) -> FrameData:
        faces: List[FaceDetection] = []
        characters: List[CharacterMatch] = []
        # 1. Use pre-computed matches for faces
        if self.face_embeddings and preds:
            for face_idx in face_indices:
                x1, y1, x2, y2 = all_face_coords[face_idx]
                matches = batch_matches[face_idx]
                candidates: List[CharacterCandidate] = []
                for char_id, similarity in matches:
                    candidates.append(CharacterCandidate(
                        character_id=char_id,
                        confidence=round(similarity, 3),
                    ))
                    self.detected_characters.add(char_id)
                # If no matches, mark as unknown with best similarity
                if not candidates and all_face_embs is not None:
                    all_matches = find_all_matches(
                        all_face_embs[face_idx],
                        self.face_embeddings,
                        0.0,
                        device=self.processor.device,
                    )
                    if all_matches:
                        candidates.append(CharacterCandidate(
                            character_id="unknown",
                            confidence=round(all_matches[0][1], 3),
                        ))
                faces.append(FaceDetection(
                    coordinates=Coordinates(
                        top=y1, right=x2, bottom=y2, left=x1,
                        width=x2-x1, height=y2-y1,
                    ),
                    candidates=candidates,
                ))
        # 2. Use pre-computed full-frame matches
        if frame_matches:
            for char_id, similarity in frame_matches:
                characters.append(CharacterMatch(
                    character_id=char_id,
                    confidence=round(similarity, 3),
                    match_type="full_image",
                ))
                self.detected_characters.add(char_id)
        return FrameData(frame_number=0, faces=faces, characters=characters)

    def _print_timing_summary(self):
        import numpy as np
        def stats(arr):
            arr = np.array(arr)
            if len(arr) == 0:
                return "n/a"
            return f"avg {arr.mean():.3f}s, min {arr.min():.3f}s, max {arr.max():.3f}s, n={len(arr)}"
        print("\n--- Processing Timing Statistics ---")
        print(f"Detection:        {stats(self.detect_timings)}")
        print(f"Frame Embedding:  {stats(self.frame_emb_timings)}")
        print(f"Frame Matching:   {stats(self.frame_match_timings)}")
        print(f"Cropping:         {stats(self.crop_timings)}")
        print(f"Face Embedding:   {stats(self.face_emb_timings)}")
        print(f"Face Matching:    {stats(self.face_match_timings)}")
        print(f"Organizing:       {stats(self.organize_timings)}")
        print(f"Batch Total:      {stats(self.batch_total_timings)}")
        print("------------------------------------\n")

    def get_results(self) -> VideoFaceRecognitionResult:
        from ..models.video import VideoProcessingStatistics
        def stat_entry(arr):
            arr = np.array(arr)
            return {
                'avg': float(arr.mean()) if len(arr) else 0.0,
                'min': float(arr.min()) if len(arr) else 0.0,
                'max': float(arr.max()) if len(arr) else 0.0,
                'n': int(len(arr)),
            }
        statistics = VideoProcessingStatistics(
            detection=stat_entry(self.detect_timings),
            frame_embedding=stat_entry(self.frame_emb_timings),
            frame_matching=stat_entry(self.frame_match_timings),
            cropping=stat_entry(self.crop_timings),
            face_embedding=stat_entry(self.face_emb_timings),
            face_matching=stat_entry(self.face_match_timings),
            organizing=stat_entry(self.organize_timings),
            batch_total=stat_entry(self.batch_total_timings),
        )
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
            statistics=statistics,
        )

    def save_results(self) -> Path:
        SAMPLE_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = FACE_DETECTIONS_FILENAME_PATTERN.format(timestamp=timestamp)
        self.output_path = SAMPLE_VIDEO_DIR / filename
        results = self.get_results()
        with open(self.output_path, "w") as f:
            f.write(results.model_dump_json(indent=2))
        return self.output_path

    def print_summary(self) -> None:
        duration_minutes = self.video_duration_seconds / 60
        output_str = f" Saved report at {self.output_path}" if self.output_path else ""
        print(
            f"Processed {self.total_frames_processed} frames "
            f"({duration_minutes:.1f} min video) in {self.processing_duration_seconds:.1f}s, "
            f"recognized {len(self.detected_characters)} characters.{output_str}",
            file=sys.stderr
        )
        # Print timing statistics summary at the end
        self._print_timing_summary()
