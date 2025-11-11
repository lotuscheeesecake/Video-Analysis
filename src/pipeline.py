# src/pipeline.py
"""
Complete video processing pipeline for feature extraction
Includes: video decoding, audio extraction, OCR, object detection, tracking
"""

import torch
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import librosa
from PIL import Image
import pytesseract
from collections import defaultdict

try:
    from decord import VideoReader, cpu
except ImportError:
    print("Warning: decord not installed")

try:
    import whisper
except ImportError:
    print("Warning: whisper not installed for ASR")


class VideoProcessor:
    """Handles video decoding and frame sampling"""
    
    def __init__(self, fps_sample=5, max_frames=128):
        self.fps_sample = fps_sample
        self.max_frames = max_frames
    
    def extract_frames(self, video_path: str) -> Tuple[np.ndarray, dict]:
        """
        Extract frames from video
        Returns: frames (T, H, W, C), metadata dict
        """
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        fps = vr.get_avg_fps()
        duration = total_frames / fps
        
        # Sample frames at target fps
        target_frames = int(duration * self.fps_sample)
        target_frames = min(target_frames, self.max_frames)
        
        if total_frames <= target_frames:
            indices = list(range(total_frames))
        else:
            indices = np.linspace(0, total_frames-1, target_frames, dtype=int)
        
        frames = vr.get_batch(indices).asnumpy()
        
        metadata = {
            'total_frames': total_frames,
            'fps': fps,
            'duration': duration,
            'sampled_frames': len(indices),
            'resolution': (frames.shape[2], frames.shape[1])
        }
        
        return frames, metadata
    
    def detect_shot_boundaries(self, frames: np.ndarray, threshold=30.0) -> List[int]:
        """Detect scene changes using frame difference"""
        boundaries = [0]
        
        for i in range(1, len(frames)):
            prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            diff = np.mean(np.abs(curr_gray.astype(float) - prev_gray.astype(float)))
            
            if diff > threshold:
                boundaries.append(i)
        
        boundaries.append(len(frames))
        return boundaries


class AudioProcessor:
    """Extract and process audio features"""
    
    def __init__(self, sr=16000, n_mels=128, hop_length=512):
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length
    
    def extract_audio(self, video_path: str) -> Tuple[np.ndarray, int]:
        """Extract audio from video using librosa"""
        try:
            audio, sr = librosa.load(video_path, sr=self.sr, mono=True)
            return audio, sr
        except Exception as e:
            print(f"Audio extraction failed: {e}")
            return None, None
    
    def compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Compute mel-spectrogram for neural network input"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_mels=self.n_mels,
            hop_length=self.hop_length
        )
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def transcribe_speech(self, audio: np.ndarray, model_name="base") -> Dict:
        """Use Whisper for ASR"""
        try:
            model = whisper.load_model(model_name)
            result = model.transcribe(audio)
            return {
                'text': result['text'],
                'segments': result['segments'],
                'language': result['language']
            }
        except Exception as e:
            print(f"ASR failed: {e}")
            return {'text': '', 'segments': [], 'language': 'unknown'}


class TextProcessor:
    """OCR and text extraction from frames"""
    
    def __init__(self):
        # Configure tesseract if needed
        # pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
        pass
    
    def extract_text_from_frame(self, frame: np.ndarray) -> str:
        """Run OCR on a single frame"""
        try:
            pil_img = Image.fromarray(frame)
            text = pytesseract.image_to_string(pil_img)
            return text.strip()
        except Exception as e:
            print(f"OCR failed: {e}")
            return ""
    
    def extract_text_from_video(self, frames: np.ndarray, sample_every=10) -> List[Tuple[int, str]]:
        """Extract text from sampled frames"""
        texts = []
        for i in range(0, len(frames), sample_every):
            text = self.extract_text_from_frame(frames[i])
            if text:
                texts.append((i, text))
        return texts


class ObjectDetectionTracker:
    """
    Object detection and tracking
    Uses YOLOv5/YOLOv8 for detection and simple IoU tracking
    """
    
    def __init__(self, model_name='yolov5s', conf_threshold=0.5):
        self.conf_threshold = conf_threshold
        # Load YOLO model (requires ultralytics or torch.hub)
        try:
            self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            self.model = None
    
    def detect_objects(self, frames: np.ndarray) -> List[Dict]:
        """
        Detect objects in frames
        Returns list of detections per frame
        """
        if self.model is None:
            return []
        
        detections = []
        for i, frame in enumerate(frames):
            results = self.model(frame)
            pred = results.pandas().xyxy[0]  # Bounding boxes
            
            frame_detections = {
                'frame_idx': i,
                'boxes': pred[['xmin', 'ymin', 'xmax', 'ymax']].values,
                'scores': pred['confidence'].values,
                'classes': pred['class'].values,
                'labels': pred['name'].values
            }
            detections.append(frame_detections)
        
        return detections
    
    def track_objects(self, detections: List[Dict], iou_threshold=0.5) -> Dict[int, List]:
        """
        Simple IoU-based tracking
        Returns dict mapping track_id to list of (frame_idx, box)
        """
        tracks = defaultdict(list)
        next_track_id = 0
        active_tracks = {}  # track_id -> last box
        
        for det in detections:
            frame_idx = det['frame_idx']
            boxes = det['boxes']
            
            if len(active_tracks) == 0:
                # Initialize tracks
                for box in boxes:
                    tracks[next_track_id].append((frame_idx, box))
                    active_tracks[next_track_id] = box
                    next_track_id += 1
            else:
                # Match boxes to existing tracks
                matched = set()
                for box in boxes:
                    best_iou = 0
                    best_track = None
                    
                    for track_id, last_box in active_tracks.items():
                        iou = self._compute_iou(box, last_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_track = track_id
                    
                    if best_iou > iou_threshold:
                        tracks[best_track].append((frame_idx, box))
                        active_tracks[best_track] = box
                        matched.add(best_track)
                    else:
                        # New track
                        tracks[next_track_id].append((frame_idx, box))
                        active_tracks[next_track_id] = box
                        next_track_id += 1
        
        return dict(tracks)
    
    @staticmethod
    def _compute_iou(box1, box2):
        """Compute IoU between two boxes [xmin, ymin, xmax, ymax]"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0


class VideoPipeline:
    """
    Main pipeline orchestrating all processing steps
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor()
        self.text_processor = TextProcessor()
        self.object_detector = ObjectDetectionTracker()
    
    def process_video(self, video_path: str) -> Dict:
        """
        Complete video analysis pipeline
        Returns comprehensive feature dictionary
        """
        print(f"Processing video: {video_path}")
        
        # 1. Extract frames
        print("Extracting frames...")
        frames, video_meta = self.video_processor.extract_frames(video_path)
        
        # 2. Detect shot boundaries
        print("Detecting shots...")
        shot_boundaries = self.video_processor.detect_shot_boundaries(frames)
        
        # 3. Extract audio
        print("Extracting audio...")
        audio, sr = self.audio_processor.extract_audio(video_path)
        mel_spec = None
        transcript = None
        if audio is not None:
            mel_spec = self.audio_processor.compute_mel_spectrogram(audio)
            transcript = self.audio_processor.transcribe_speech(audio)
        
        # 4. Extract text (OCR)
        print("Running OCR...")
        ocr_results = self.text_processor.extract_text_from_video(frames)
        
        # 5. Object detection
        print("Detecting objects...")
        detections = self.object_detector.detect_objects(frames)
        
        # 6. Object tracking
        print("Tracking objects...")
        tracks = self.object_detector.track_objects(detections)
        
        # Compile results
        results = {
            'video_metadata': video_meta,
            'frames': frames,  # (T, H, W, C)
            'shot_boundaries': shot_boundaries,
            'audio': {
                'waveform': audio,
                'sample_rate': sr,
                'mel_spectrogram': mel_spec,
                'transcript': transcript
            },
            'ocr': ocr_results,
            'detections': detections,
            'tracks': tracks
        }
        
        print("Processing complete!")
        return results
    
    def extract_features_for_model(self, results: Dict) -> Dict[str, torch.Tensor]:
        """
        Convert pipeline results to model-ready tensors
        """
        frames = results['frames']
        mel_spec = results['audio']['mel_spectrogram']
        
        # Prepare video tensor
        # Normalize and convert to tensor (T, C, H, W)
        video_tensor = torch.from_numpy(frames).float() / 255.0
        video_tensor = video_tensor.permute(0, 3, 1, 2)  # (T, C, H, W)
        
        # Resize to model input size
        video_tensor = torch.nn.functional.interpolate(
            video_tensor,
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        )
        
        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        video_tensor = (video_tensor - mean) / std
        
        # Prepare audio tensor
        audio_tensor = None
        if mel_spec is not None:
            audio_tensor = torch.from_numpy(mel_spec).float().unsqueeze(0)  # (1, freq, time)
        
        # Prepare text embedding (placeholder - use actual text encoder)
        text_tensor = None
        if results['audio']['transcript']:
            # In practice, use sentence transformer or similar
            text_tensor = torch.randn(512)  # Placeholder
        
        return {
            'video': video_tensor.unsqueeze(0),  # (1, T, C, H, W)
            'audio': audio_tensor.unsqueeze(0) if audio_tensor is not None else None,
            'text': text_tensor.unsqueeze(0) if text_tensor is not None else None
        }


# Example usage function
def process_and_analyze(video_path: str, model, device='cuda'):
    """
    Complete end-to-end processing and analysis
    """
    pipeline = VideoPipeline()
    
    # Process video through pipeline
    results = pipeline.process_video(video_path)
    
    # Extract features for model
    model_inputs = pipeline.extract_features_for_model(results)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        video = model_inputs['video'].to(device)
        audio = model_inputs['audio'].to(device) if model_inputs['audio'] is not None else None
        text = model_inputs['text'].to(device) if model_inputs['text'] is not None else None
        
        outputs = model(video, audio_spec=audio, text_embed=text)
    
    # Combine neural outputs with pipeline results
    final_results = {
        'pipeline_results': results,
        'model_predictions': outputs,
        'metadata': results['video_metadata']
    }
    
    return final_results