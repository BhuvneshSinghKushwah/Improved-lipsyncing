# Wav2Lip Enhanced

An enhanced fork of [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) with **Apple Silicon (MPS) support**, **GFPGAN face enhancement**, and **improved blending** for higher quality lip-synced videos.

## What's New in This Fork

### Apple Silicon (MPS) Support
- Full support for Apple M1/M2/M3 chips via Metal Performance Shaders (MPS)
- Automatic device detection (CUDA → MPS → CPU)
- Memory optimizations for MPS to prevent OOM errors
- Reduced default batch sizes for better stability on Mac

### GFPGAN Face Enhancement
- Integrated [GFPGAN](https://github.com/TencentARC/GFPGAN) for real-time face restoration
- Dramatically improves visual quality of generated faces
- Supports GFPGANv1.3 and GFPGANv1.4 models
- Optional temporal blending to reduce flicker between frames

### Feathered Blending
- Smooth gradient blending at mouth region edges
- Eliminates harsh "pasted on" look of original Wav2Lip
- Configurable feather amount for fine-tuning

### Performance Optimizations
- Face detection interval option (skip frames and interpolate)
- GFPGAN enhancement interval (enhance every Nth frame)
- Singleton pattern for face detector to reduce memory usage
- Aggressive memory cleanup between processing stages

### Bug Fixes
- Fixed `librosa.filters.mel` API for newer librosa versions
- Fixed NaN handling in face detection
- Fixed NMS overflow issues with float64 casting
- Added missing torch import in SFD detector
- Fixed `torchvision.transforms.functional_tensor` deprecation

---

## Installation

### Prerequisites
- Python 3.8+
- ffmpeg: `brew install ffmpeg` (Mac) or `sudo apt-get install ffmpeg` (Linux)

### Setup

```bash
# Clone this repository
git clone https://github.com/YOUR_USERNAME/Wav2Lip.git
cd Wav2Lip

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download face detection model
mkdir -p face_detection/detection/sfd
wget "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" -O face_detection/detection/sfd/s3fd.pth

# Download Wav2Lip checkpoints (place in checkpoints/ folder)
# Get from: https://github.com/Rudrabha/Wav2Lip#getting-the-weights
```

### GFPGAN Setup (Optional but Recommended)

```bash
pip install gfpgan

# If you encounter torchvision compatibility issues, run:
sed -i '' 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' \
  .venv/lib/python*/site-packages/basicsr/data/degradations.py
```

---

## Usage

### Basic Inference

```bash
python inference.py \
  --checkpoint_path checkpoints/wav2lip_gan.pth \
  --face input_video.mp4 \
  --audio input_audio.wav \
  --outfile results/output.mp4
```

### With GFPGAN Enhancement (Best Quality)

```bash
python inference.py \
  --checkpoint_path checkpoints/wav2lip_gan.pth \
  --face input_video.mp4 \
  --audio input_audio.wav \
  --enhance \
  --feather_amount 5 \
  --outfile results/output_enhanced.mp4
```

### Fast Mode (For Long Videos)

```bash
python inference.py \
  --checkpoint_path checkpoints/wav2lip_gan.pth \
  --face input_video.mp4 \
  --audio input_audio.wav \
  --enhance \
  --enhance_interval 3 \
  --face_det_interval 5 \
  --outfile results/output_fast.mp4
```

---

## New Command Line Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--enhance` | False | Enable GFPGAN face enhancement |
| `--enhancer_model` | GFPGANv1.4 | GFPGAN model version (GFPGANv1.3 or GFPGANv1.4) |
| `--enhance_interval` | 1 | Apply GFPGAN every N frames (higher = faster) |
| `--enhance_blend` | 0.0 | Temporal blending (0.0-0.9) to reduce flicker |
| `--no_feather` | False | Disable feathered blending |
| `--feather_amount` | 3 | Edge feathering amount (higher = smoother blend) |
| `--face_det_interval` | 1 | Face detection every N frames (higher = faster) |
| `--face_det_batch_size` | 4 | Batch size for face detection (reduced for MPS) |

---

## Quality vs Speed Tradeoffs

| Mode | Command Flags | Quality | Speed |
|------|---------------|---------|-------|
| **Best Quality** | `--enhance --feather_amount 5` | Excellent | Slow |
| **Balanced** | `--enhance --enhance_interval 2` | Very Good | Medium |
| **Fast** | `--enhance_interval 5 --face_det_interval 10` | Good | Fast |
| **Original** | `--no_feather` (no --enhance) | Basic | Fastest |

---

## Checkpoints

| Model | Description | Link |
|-------|-------------|------|
| Wav2Lip | Highly accurate lip-sync | [Download](https://drive.google.com/drive/folders/153HLrqlBNxzZcHi17PEvP09kkAfzRshM) |
| Wav2Lip + GAN | Better visual quality | [Download](https://drive.google.com/file/d/15G3U08c8xsCkOqQxE38Z2XXDnPcOptNk/view) |

---

## Changes Summary

### Modified Files

- **`inference.py`** - Added GFPGAN integration, feathered blending, MPS support, memory optimizations
- **`requirements.txt`** - Updated dependencies, added gfpgan
- **`audio.py`** - Fixed librosa API compatibility
- **`face_detection/api.py`** - Added NaN handling in detections
- **`face_detection/detection/core.py`** - Added MPS device support
- **`face_detection/detection/sfd/bbox.py`** - Fixed NMS overflow with float64
- **`face_detection/detection/sfd/sfd_detector.py`** - Added missing torch import

---

## Troubleshooting

### OOM Errors on Mac (MPS)
Reduce batch sizes:
```bash
python inference.py ... --wav2lip_batch_size 8 --face_det_batch_size 2
```

### GFPGAN Import Error
If you see `ModuleNotFoundError: torchvision.transforms.functional_tensor`:
```bash
# Fix the basicsr compatibility issue
sed -i '' 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' \
  .venv/lib/python*/site-packages/basicsr/data/degradations.py
```

### Face Not Detected
- Try `--pads 0 20 0 0` to include more chin area
- Use `--resize_factor 2` for high-resolution videos
- Use `--nosmooth` if face detection is jittery

---

## Credits

### Original Wav2Lip
This project is based on [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) by Rudrabha Mukhopadhyay et al.

**Paper:** [A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild](http://arxiv.org/abs/2008.10010) (ACM Multimedia 2020)

```bibtex
@inproceedings{10.1145/3394171.3413532,
  author = {Prajwal, K R and Mukhopadhyay, Rudrabha and Namboodiri, Vinay P. and Jawahar, C.V.},
  title = {A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild},
  year = {2020},
  booktitle = {Proceedings of the 28th ACM International Conference on Multimedia},
  pages = {484–492},
  series = {MM '20}
}
```

### GFPGAN
Face enhancement powered by [GFPGAN](https://github.com/TencentARC/GFPGAN) by Tencent ARC Lab.

### Face Detection
Face detection from [face_alignment](https://github.com/1adrianb/face-alignment) repository.

---

## License

This repository is for **personal/research/non-commercial purposes only**. For commercial use, contact the original Wav2Lip authors at rudrabha@synclabs.so or prajwal@synclabs.so.

Commercial API available at: https://synclabs.so/
