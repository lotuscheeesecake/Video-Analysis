# Video Analysis & Semantic Search System

A state-of-the-art deep learning system for video understanding, keyword prediction, and semantic search. Built with PyTorch, Vision Transformers, and FAISS for efficient video retrieval.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## Features

- **Multi-Modal Video Analysis**: Visual, audio, and text processing
- **Semantic Search**: Find videos using natural language queries or filters
- **Keyword Prediction**: Automatic tagging with custom labels
- **Object Detection**: Detect and track objects across video frames
- **Audio Processing**: Speech recognition and audio feature extraction
- **OCR**: Extract text from video frames
- **Efficient Indexing**: Fast retrieval with FAISS vector search
- **Pre-trained Models**: Built on Vision Transformers and modern architectures

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/video-analysis-system.git
cd video-analysis-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run quick setup
chmod +x quickstart.sh
./quickstart.sh  # On Windows: quickstart.bat
```

### System Requirements

- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended)
- **RAM**: 16GB minimum
- **Storage**: 10GB+ for models and indices
- **FFmpeg**: Required for video/audio processing

### Basic Usage

**1. Prepare Your Data**

```bash
# Put training videos in data/videos/
cp your_videos/*.mp4 data/videos/

# Edit labels
nano vocab/labels.json

# Create training manifest
nano data/manifest.csv
```

**2. Train the Model**

```bash
python src/train.py --config configs/config.yaml
```

**3. Build Search Index**

```bash
python scripts/build_index.py \
    --video_dir ./videos \
    --model_path ./checkpoints/best.pt \
    --output_dir ./video_index
```

**4. Search Videos**

```python
from search_indexing import VideoSemanticIndex, MultimodalQueryEngine

# Load index
index = VideoSemanticIndex(embedding_dim=512)
index.load('./video_index')

# Search
query_engine = MultimodalQueryEngine(index)
results = query_engine.query(
    required_keywords=['person', 'running'],
    min_duration=30,
    k=10
)

for metadata, score in results:
    print(f"{metadata.video_id}: {score:.3f}")
```

## Project Structure

```
video-analysis-system/
├── configs/
│   └── config.yaml              # Configuration file
├── data/
│   ├── manifest.csv             # Training manifest
│   └── videos/                  # Training videos
├── videos/                      # Videos for indexing
├── vocab/
│   └── labels.json              # Keyword labels
├── src/
│   ├── datasets.py              # Dataset loaders
│   ├── models.py                # Model architectures
│   ├── train.py                 # Training script
│   ├── infer.py                 # Inference script
│   ├── metrics.py               # Evaluation metrics
│   ├── utils.py                 # Utility functions
│   ├── pipeline.py              # Video processing pipeline
│   └── search_indexing.py       # Search & indexing
├── scripts/
│   └── build_index.py           # Index building script
├── checkpoints/                 # Saved models
├── video_index/                 # Search index
└── requirements.txt             # Python dependencies
```

## Model Architecture

The system uses a multi-modal architecture combining:

- **Visual Features**: Vision Transformer (ViT) or ResNet backbone
- **Temporal Modeling**: Transformer encoder or SlowFast networks
- **Audio Processing**: Mel-spectrogram CNN + Whisper ASR
- **Text Processing**: OCR (Tesseract) + sentence embeddings
- **Fusion**: Multi-modal transformer for cross-modal attention
- **Heads**: Object detection, action recognition, scene classification, keyword prediction

### Supported Backbones

- Vision Transformer (ViT): `vit_base_patch16_224`, `vit_large_patch16_224`
- ResNet: `resnet50`, `resnet101`
- EfficientNet: `efficientnet_b0` through `efficientnet_b7`

## Training

### Configuration

Edit `configs/config.yaml`:

```yaml
dataset:
  num_frames: 32        # Frames per video
  image_size: 224       # Image resolution

model:
  visual_backbone: "vit_base_patch16_224"
  temporal_layers: 6
  num_keywords: 300     # Your custom labels

training:
  batch_size: 4
  epochs: 20
  lr: 5e-5
  device: "cuda"
```

### Training Commands

```bash
# Basic training
python src/train.py --config configs/config.yaml

# Resume from checkpoint
python src/train.py --config configs/config.yaml --resume checkpoints/ckpt_epoch_10.pt

# Train on CPU (slow)
python src/train.py --config configs/config.yaml --device cpu
```

### Monitoring

Training outputs:
- Loss curves and metrics
- Checkpoints: `checkpoints/ckpt_epoch_*.pt`
- Best model: `checkpoints/best.pt`
- Logs: `logs/`

## Semantic Search

### Building the Index

```bash
python scripts/build_index.py \
    --video_dir ./videos \
    --model_path ./checkpoints/best.pt \
    --output_dir ./video_index \
    --device cuda
```

### Search Examples

**By Keywords:**
```python
results = query_engine.query(
    required_keywords=['cooking', 'kitchen'],
    k=10
)
```

**With Duration Filter:**
```python
results = query_engine.query(
    required_keywords=['running'],
    min_duration=60,
    max_duration=300,
    k=5
)
```

**Complex Query:**
```python
results = query_engine.query(
    required_keywords=['person', 'outdoor'],
    required_objects=['car', 'tree'],
    required_actions=['walking'],
    min_duration=30,
    k=10
)
```

## Inference

### Single Video Analysis

```bash
python src/infer.py \
    --config configs/config.yaml \
    --ckpt checkpoints/best.pt \
    --video videos/test_video.mp4
```

**Output:**
```
Predicted keywords (label, score):
person: 0.9234
running: 0.8456
outdoor: 0.7823
happy: 0.6712
car: 0.5891
```

### Batch Processing

```python
from pipeline import VideoPipeline

pipeline = VideoPipeline()
results = pipeline.process_video('video.mp4')

# Access features
print(results['video_metadata'])
print(results['detections'])
print(results['audio']['transcript'])
```

## Data Format

### Labels JSON (`vocab/labels.json`)

```json
[
  "person",
  "car",
  "running",
  "cooking",
  "outdoor",
  "happy"
]
```

### Training Manifest (`data/manifest.csv`)

```csv
video_path,label_indices
data/videos/video1.mp4,"0,2,4"
data/videos/video2.mp4,"1,3,5"
data/videos/video3.mp4,"0,1,2"
```

**Label indices** correspond to positions in `labels.json` (0-indexed).

## Advanced Usage

### Custom Model

```python
from models import VideoAnalysisModel

model = VideoAnalysisModel(
    num_object_classes=80,
    num_action_classes=400,
    num_keywords=300,
    visual_backbone='vit_large_patch16_224',
    use_slowfast=True,
    fusion_dim=1024
)
```

### Custom Pipeline

```python
from pipeline import VideoPipeline

class MyPipeline(VideoPipeline):
    def custom_processing(self, frames):
        # Your custom logic
        return processed_frames

pipeline = MyPipeline()
results = pipeline.process_video('video.mp4')
```

### Export Search Results

```python
import json

# Search and export
results = query_engine.query(required_keywords=['cooking'], k=20)

output = [{
    'video_id': m.video_id,
    'path': m.video_path,
    'score': float(s),
    'duration': m.duration,
    'keywords': m.keywords
} for m, s in results]

with open('results.json', 'w') as f:
    json.dump(output, f, indent=2)
```

## Troubleshooting

### CUDA Out of Memory

```yaml
# Reduce batch size
training:
  batch_size: 2
  grad_accum_steps: 8
```

### Video Loading Errors

```bash
# Re-encode videos
ffmpeg -i input.mov -c:v libx264 -c:a aac output.mp4
```

### Slow Training

```yaml
# Use fewer frames
dataset:
  num_frames: 16

# Reduce model complexity
model:
  temporal_layers: 4
```

### Import Errors

```bash
# Ensure you're in project root
cd video-analysis-system
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- [TIMM](https://github.com/huggingface/pytorch-image-models) - PyTorch Image Models
- [Decord](https://github.com/dmlc/decord) - Efficient video loading
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search
- [Whisper](https://github.com/openai/whisper) - Speech recognition
- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLOv8 object detection

## Contact

Abdul Haleem Cheema - (www.linkedin.com/in/abdul-haleem-cheema) - abdul.haleem.cheema@gmail.com

Project Literature: [https://github.com/yourusername/video-analysis-system](https://github.com/yourusername/video-analysis-system)

⭐ **Star this repo if you find it helpful!** ⭐
