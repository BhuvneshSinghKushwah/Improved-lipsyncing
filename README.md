# Wav2Lip Enhanced

> **Note:** This project is still under active development. There are a few quirks to be fixed, but the final goal is to provide seamless, high-quality lip-synced video generation with native Apple Silicon support and enhanced face restoration.

An enhanced fork of [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) with **Apple Silicon (MPS) support**, **Real-ESRGAN/GFPGAN face enhancement**, and **improved blending** for higher quality lip-synced videos.

## Sample Outputs

| Input Image | Enhancer | Output Video |
|:-----------:|:--------:|:------------:|
| ![input_image](https://github.com/user-attachments/assets/3fd131f1-7f66-49d3-b523-1de2df0e4d1c) | No Enhancement | <video src="https://github.com/user-attachments/assets/dca7cf62-54e8-44ab-8fc1-fc2a4dabf16d" width="200"></video> |
| ![input_image](https://github.com/user-attachments/assets/3fd131f1-7f66-49d3-b523-1de2df0e4d1c) | GFPGAN v1.4 | <video src="https://github.com/user-attachments/assets/4d2d89ae-d9ef-4da7-930f-c97068367f79" width="200"></video> |
| ![input_image](https://github.com/user-attachments/assets/3fd131f1-7f66-49d3-b523-1de2df0e4d1c) | Real-ESRGAN x2plus | <video src="https://github.com/user-attachments/assets/a8800b4f-4006-4848-9dce-c417fe2b1878" width="200"></video> |
| ![input_image](https://github.com/user-attachments/assets/3fd131f1-7f66-49d3-b523-1de2df0e4d1c) | Real-ESRGAN x4plus | <video src="https://github.com/user-attachments/assets/109c8fea-8daf-4e12-b0b9-b875584766d6" width="200"></video> |

## What's New in This Fork

### Real-ESRGAN Face Enhancement (New!)
- Integrated [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) as the default face enhancer
- Cleaner textures and more natural skin detail compared to GFPGAN
- Multiple models available:
  - `RealESRGAN_x2plus` - Fast, good quality (default)
  - `RealESRGAN_x4plus` - Higher quality, slower
  - `realesr-general-x4v3` - Lightweight alternative
- Automatic fallback to GFPGAN if Real-ESRGAN unavailable

### Apple Silicon (MPS) Support
- Full support for Apple M1/M2/M3/M4 chips via Metal Performance Shaders (MPS)
- Automatic device detection (CUDA → MPS → CPU)
- Smart chip detection with recommended batch sizes
- `--device-info` flag to check your hardware capabilities
- `--auto-batch` flag to automatically configure optimal batch sizes
- MLX framework detection for future native Apple Silicon optimizations

### GFPGAN Face Enhancement
- Integrated [GFPGAN](https://github.com/TencentARC/GFPGAN) as alternative enhancer
- Supports GFPGANv1.3 and GFPGANv1.4 models
- Use `--enhancer gfpgan` to switch from Real-ESRGAN

### Feathered Blending
- Smooth gradient blending at mouth region edges
- Eliminates harsh "pasted on" look of original Wav2Lip
- Configurable feather amount for fine-tuning

### Performance Optimizations
- Face detection interval option (skip frames and interpolate)
- Enhancement interval (enhance every Nth frame)
- Temporal blending to reduce flicker between frames
- Singleton pattern for face detector to reduce memory usage
- Aggressive memory cleanup between processing stages

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

### Check Your Device

```bash
python inference.py --device-info
```

---

## Usage

### Basic Inference (No Enhancement)

```bash
python inference.py \
  --checkpoint_path checkpoints/wav2lip_gan.pth \
  --face input_video.mp4 \
  --audio input_audio.wav \
  --outfile results/output.mp4
```

### With Real-ESRGAN Enhancement (Recommended)

```bash
python inference.py \
  --checkpoint_path checkpoints/wav2lip_gan.pth \
  --face input_video.mp4 \
  --audio input_audio.wav \
  --enhance \
  --outfile results/output_enhanced.mp4
```

### With Real-ESRGAN x4plus (Best Quality)

```bash
python inference.py \
  --checkpoint_path checkpoints/wav2lip_gan.pth \
  --face input_video.mp4 \
  --audio input_audio.wav \
  --enhance \
  --enhancer_model RealESRGAN_x4plus \
  --outfile results/output_x4.mp4
```

### With GFPGAN Enhancement

```bash
python inference.py \
  --checkpoint_path checkpoints/wav2lip_gan.pth \
  --face input_video.mp4 \
  --audio input_audio.wav \
  --enhance \
  --enhancer gfpgan \
  --enhancer_model GFPGANv1.4 \
  --outfile results/output_gfpgan.mp4
```

### Apple Silicon Optimized

```bash
python inference.py \
  --checkpoint_path checkpoints/wav2lip_gan.pth \
  --face input_video.mp4 \
  --audio input_audio.wav \
  --enhance \
  --auto-batch \
  --outfile results/output.mp4
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

## Command Line Options

### Enhancement Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--enhance` | False | Enable face enhancement |
| `--enhancer` | realesrgan | Enhancer to use: `realesrgan` or `gfpgan` |
| `--enhancer_model` | RealESRGAN_x2plus | Model to use (see below) |
| `--enhance_interval` | 1 | Apply enhancement every N frames |
| `--enhance_blend` | 0.0 | Temporal blending (0.0-0.9) to reduce flicker |

### Available Enhancement Models

| Enhancer | Model | Quality | Speed |
|----------|-------|---------|-------|
| Real-ESRGAN | `RealESRGAN_x2plus` | Good | Fast |
| Real-ESRGAN | `RealESRGAN_x4plus` | Best | Slow |
| Real-ESRGAN | `realesr-general-x4v3` | Good | Fast |
| GFPGAN | `GFPGANv1.4` | Good | Medium |
| GFPGAN | `GFPGANv1.3` | Good | Medium |

### Blending Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--no_feather` | False | Disable feathered blending |
| `--feather_amount` | 3 | Edge feathering amount |

### Performance Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--face_det_interval` | 1 | Face detection every N frames |
| `--face_det_batch_size` | 4 | Batch size for face detection |
| `--wav2lip_batch_size` | 128 | Batch size for Wav2Lip |
| `--auto-batch` | False | Auto-configure batch sizes for your device |
| `--device-info` | False | Print device info and exit |

---

## Quality vs Speed Tradeoffs

| Mode | Command Flags | Quality | Speed |
|------|---------------|---------|-------|
| **Best Quality** | `--enhance --enhancer_model RealESRGAN_x4plus` | Excellent | Slow |
| **Balanced** | `--enhance` (default x2plus) | Very Good | Medium |
| **Fast Enhanced** | `--enhance --enhance_interval 3` | Good | Fast |
| **GFPGAN** | `--enhance --enhancer gfpgan` | Good | Medium |
| **No Enhancement** | (no --enhance flag) | Basic | Fastest |

---

## Checkpoints

| Model | Description | Link |
|-------|-------------|------|
| Wav2Lip | Highly accurate lip-sync | [Download](https://drive.google.com/drive/folders/153HLrqlBNxzZcHi17PEvP09kkAfzRshM) |
| Wav2Lip + GAN | Better visual quality | [Download](https://drive.google.com/file/d/15G3U08c8xsCkOqQxE38Z2XXDnPcOptNk/view) |

---

## Troubleshooting

### OOM Errors on Mac (MPS)
Reduce batch sizes:
```bash
python inference.py ... --wav2lip_batch_size 8 --face_det_batch_size 2
```

Or use auto-batch:
```bash
python inference.py ... --auto-batch
```

### Real-ESRGAN Import Error
```bash
pip install realesrgan basicsr
```

### GFPGAN Import Error
If you see `ModuleNotFoundError: torchvision.transforms.functional_tensor`:
```bash
sed -i '' 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' \
  .venv/lib/python*/site-packages/basicsr/data/degradations.py
```

### Face Not Detected
- Try `--pads 0 20 0 0` to include more chin area
- Use `--resize_factor 2` for high-resolution videos
- Use `--nosmooth` if face detection is jittery

---

## Author

Enhanced by [Bhuvnesh Singh Kushwah](https://www.linkedin.com/in/bhuvnesh-singh-kushwah/)

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

### Real-ESRGAN
Face enhancement powered by [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) by Xintao Wang et al.

### GFPGAN
Alternative face enhancement by [GFPGAN](https://github.com/TencentARC/GFPGAN) by Tencent ARC Lab.

### Face Detection
Face detection from [face_alignment](https://github.com/1adrianb/face-alignment) repository.

### MLX
Apple's [MLX](https://github.com/ml-explore/mlx) framework for efficient ML on Apple Silicon.

---

## License

This repository is for **personal/research/non-commercial purposes only**. For commercial use, contact the original Wav2Lip authors at rudrabha@synclabs.so or prajwal@synclabs.so.

Commercial API available at: https://synclabs.so/
