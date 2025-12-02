"""
Stable Diffusion inpainting for mouth region refinement.

This module provides diffusion-based enhancement to refine the mouth region
after Wav2Lip processing, improving visual quality while preserving lip-sync accuracy.
"""

import numpy as np
from PIL import Image
import cv2

# Check for diffusion availability
DIFFUSION_AVAILABLE = False
try:
    import torch
    from diffusers import AutoPipelineForInpainting, StableDiffusionUpscalePipeline
    DIFFUSION_AVAILABLE = True
except ImportError:
    pass


# Model configurations
DIFFUSION_MODELS = {
    'sdxl': {
        'model_id': 'diffusers/stable-diffusion-xl-1.0-inpainting-0.1',
        'size': 1024,
        'type': 'inpainting',
        'description': 'SDXL Inpainting - Best quality, slower'
    },
    'sd15': {
        'model_id': 'runwayml/stable-diffusion-inpainting',
        'size': 512,
        'type': 'inpainting',
        'description': 'SD 1.5 Inpainting - Faster, good quality'
    },
    'sr3': {
        'model_id': 'stabilityai/stable-diffusion-x4-upscaler',
        'scale': 4,
        'type': 'super_resolution',
        'description': 'SR3-style 4x Super-Resolution - Sharp details, fast'
    }
}


def load_diffusion_enhancer(model_name='sdxl', device='cpu'):
    """
    Load Stable Diffusion pipeline (inpainting or super-resolution).

    Args:
        model_name: 'sdxl' (best quality), 'sd15' (faster), or 'sr3' (super-resolution)
        device: 'cuda', 'mps', or 'cpu'

    Returns:
        Tuple of (pipeline, model_type) or (None, None) if unavailable
    """
    if not DIFFUSION_AVAILABLE:
        print("Warning: Diffusion not available. Install with: pip install diffusers transformers accelerate")
        return None, None

    if model_name not in DIFFUSION_MODELS:
        print(f"Warning: Unknown diffusion model '{model_name}'. Using 'sdxl'.")
        model_name = 'sdxl'

    model_config = DIFFUSION_MODELS[model_name]
    model_id = model_config['model_id']
    model_type = model_config['type']

    print(f"Loading diffusion model: {model_name} ({model_config['description']})")
    print(f"  Model ID: {model_id}")
    print(f"  Type: {model_type}")
    print(f"  This may take a while on first run (downloading model)...")

    try:
        if model_type == 'super_resolution':
            # Load SR3-style super-resolution pipeline
            if device == 'mps':
                pipe = StableDiffusionUpscalePipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                )
                pipe = pipe.to(device)
                pipe.enable_attention_slicing()
                print(f"  Loaded SR3 with MPS optimizations (fp32, attention slicing)")
            elif device == 'cuda':
                pipe = StableDiffusionUpscalePipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                )
                pipe = pipe.to(device)
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                    print(f"  Loaded SR3 with CUDA optimizations (fp16, xformers)")
                except Exception:
                    print(f"  Loaded SR3 with CUDA optimizations (fp16)")
            else:
                pipe = StableDiffusionUpscalePipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                )
                print(f"  Loaded SR3 on CPU (slow)")
        else:
            # Load inpainting pipeline
            if device == 'mps':
                pipe = AutoPipelineForInpainting.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    use_safetensors=True
                )
                pipe = pipe.to(device)
                pipe.enable_attention_slicing()
                print(f"  Loaded with MPS optimizations (fp32, attention slicing)")

            elif device == 'cuda':
                pipe = AutoPipelineForInpainting.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    use_safetensors=True
                )
                pipe = pipe.to(device)
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                    print(f"  Loaded with CUDA optimizations (fp16, xformers)")
                except Exception:
                    print(f"  Loaded with CUDA optimizations (fp16)")

            else:
                pipe = AutoPipelineForInpainting.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    use_safetensors=True
                )
                print(f"  Loaded on CPU (slow, consider using GPU)")

        print(f"  Diffusion model loaded successfully!")
        return pipe, model_type

    except Exception as e:
        print(f"Error loading diffusion model: {e}")
        return None, None


def create_mouth_mask(frame, face_coords, expansion=1.3):
    """
    Create a mask for the mouth region based on face bounding box.

    The mouth region is approximately the lower 40% of the face,
    centered horizontally with some horizontal padding.

    Args:
        frame: Input frame (numpy array, BGR format)
        face_coords: Tuple of (y1, y2, x1, x2) face bounding box
        expansion: Factor to expand the mouth region (default 1.3)

    Returns:
        PIL Image mask (white = area to inpaint, black = keep)
    """
    h, w = frame.shape[:2]
    y1, y2, x1, x2 = face_coords

    face_height = y2 - y1
    face_width = x2 - x1

    # Mouth region: lower 40% of face
    # Start from 55% down to leave some nose/cheek area
    mouth_y1 = y1 + int(face_height * 0.55)
    mouth_y2 = y2 + int(face_height * 0.05)  # Extend slightly below face box for chin

    # Horizontal: center 60% of face width (avoid ears/cheeks)
    mouth_x1 = x1 + int(face_width * 0.2)
    mouth_x2 = x2 - int(face_width * 0.2)

    # Calculate center and dimensions
    center_y = (mouth_y1 + mouth_y2) // 2
    center_x = (mouth_x1 + mouth_x2) // 2
    half_height = int((mouth_y2 - mouth_y1) * expansion / 2)
    half_width = int((mouth_x2 - mouth_x1) * expansion / 2)

    # Apply expansion
    mouth_y1 = max(0, center_y - half_height)
    mouth_y2 = min(h, center_y + half_height)
    mouth_x1 = max(0, center_x - half_width)
    mouth_x2 = min(w, center_x + half_width)

    # Create mask (black background, white mouth region)
    mask = np.zeros((h, w), dtype=np.uint8)

    # Create elliptical mask for more natural blending
    center = ((mouth_x1 + mouth_x2) // 2, (mouth_y1 + mouth_y2) // 2)
    axes = ((mouth_x2 - mouth_x1) // 2, (mouth_y2 - mouth_y1) // 2)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    # Apply Gaussian blur for soft edges
    mask = cv2.GaussianBlur(mask, (31, 31), 11)

    # Convert to PIL Image
    mask_pil = Image.fromarray(mask)

    return mask_pil


def enhance_frame_diffusion(frame, pipe, face_coords,
                           strength=0.5, steps=25,
                           prompt="a person with natural realistic mouth and lips, high quality photo",
                           negative_prompt="blurry, distorted, unnatural, artifacts"):
    """
    Refine the mouth region using Stable Diffusion inpainting.

    Args:
        frame: Input frame (numpy array, BGR format)
        pipe: Diffusers inpainting pipeline
        face_coords: Tuple of (y1, y2, x1, x2) face bounding box
        strength: Denoising strength (0.0-1.0). Higher = more change
        steps: Number of inference steps (10-50)
        prompt: Text prompt for generation
        negative_prompt: Negative prompt to avoid

    Returns:
        Enhanced frame (numpy array, BGR format)
    """
    if pipe is None:
        return frame

    import torch

    original_h, original_w = frame.shape[:2]

    # Get model's expected size
    model_size = 1024 if 'xl' in str(type(pipe).__name__).lower() or 'xl' in str(pipe.config).lower() else 512

    # Determine model size from pipeline
    try:
        if hasattr(pipe, 'unet') and hasattr(pipe.unet, 'config'):
            sample_size = pipe.unet.config.sample_size
            model_size = sample_size * 8  # VAE downscale factor
    except Exception:
        pass

    # Clamp model size to reasonable values
    model_size = min(max(model_size, 512), 1024)

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create PIL image
    image_pil = Image.fromarray(frame_rgb)

    # Resize for diffusion model
    image_resized = image_pil.resize((model_size, model_size), Image.LANCZOS)

    # Create mouth mask (need to scale face_coords for resized image)
    scale_x = model_size / original_w
    scale_y = model_size / original_h

    y1, y2, x1, x2 = face_coords
    scaled_coords = (
        int(y1 * scale_y),
        int(y2 * scale_y),
        int(x1 * scale_x),
        int(x2 * scale_x)
    )

    # Create mask at model size
    mask_frame = np.zeros((model_size, model_size, 3), dtype=np.uint8)
    mask_pil = create_mouth_mask(mask_frame, scaled_coords, expansion=1.3)

    # Run inpainting
    try:
        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image_resized,
                mask_image=mask_pil,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=7.5,
            ).images[0]

        # Resize back to original size
        result_resized = result.resize((original_w, original_h), Image.LANCZOS)

        # Convert back to BGR numpy array
        result_np = np.array(result_resized)
        result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)

        # Get mask at original size for blending
        mask_original = mask_pil.resize((original_w, original_h), Image.LANCZOS)
        mask_np = np.array(mask_original).astype(np.float32) / 255.0
        mask_np = mask_np[:, :, np.newaxis]  # Add channel dimension

        # Blend diffusion result with original using mask
        # This ensures only the mouth region is affected
        blended = (result_bgr * mask_np + frame * (1 - mask_np)).astype(np.uint8)

        return blended

    except Exception as e:
        print(f"Warning: Diffusion enhancement failed: {e}")
        return frame


def enhance_frame_sr3(frame, pipe, face_coords, steps=25,
                      prompt="high quality detailed face, sharp features, realistic skin"):
    """
    Enhance the face region using SR3-style super-resolution.

    Args:
        frame: Input frame (numpy array, BGR format)
        pipe: Diffusers super-resolution pipeline
        face_coords: Tuple of (y1, y2, x1, x2) face bounding box
        steps: Number of inference steps
        prompt: Text prompt for generation

    Returns:
        Enhanced frame (numpy array, BGR format)
    """
    if pipe is None:
        return frame

    import torch

    y1, y2, x1, x2 = face_coords
    original_h, original_w = frame.shape[:2]

    # Expand face region slightly for better context
    expand = 20
    y1_exp = max(0, y1 - expand)
    y2_exp = min(original_h, y2 + expand)
    x1_exp = max(0, x1 - expand)
    x2_exp = min(original_w, x2 + expand)

    # Extract face region
    face_region = frame[y1_exp:y2_exp, x1_exp:x2_exp]
    face_h, face_w = face_region.shape[:2]

    # Convert BGR to RGB
    face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)

    # SR3 expects low-res input, so we downsample first then upsample
    # Downsample to create low-res version
    low_res_size = (max(64, face_w // 4), max(64, face_h // 4))
    low_res = cv2.resize(face_rgb, low_res_size, interpolation=cv2.INTER_AREA)
    low_res_pil = Image.fromarray(low_res)

    try:
        with torch.no_grad():
            # Run super-resolution
            result = pipe(
                prompt=prompt,
                image=low_res_pil,
                num_inference_steps=steps,
                guidance_scale=7.5,
            ).images[0]

        # Resize result to match original face region size
        result_resized = result.resize((face_w, face_h), Image.LANCZOS)

        # Convert back to BGR
        result_np = np.array(result_resized)
        result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)

        # Create smooth blending mask for edges
        blend_mask = np.ones((face_h, face_w), dtype=np.float32)
        feather = 10
        for i in range(feather):
            alpha = (i + 1) / (feather + 1)
            blend_mask[i, :] = alpha
            blend_mask[face_h - 1 - i, :] = alpha
            blend_mask[:, i] = np.minimum(blend_mask[:, i], alpha)
            blend_mask[:, face_w - 1 - i] = np.minimum(blend_mask[:, face_w - 1 - i], alpha)
        blend_mask = blend_mask[:, :, np.newaxis]

        # Blend with original
        blended_face = (result_bgr * blend_mask + face_region * (1 - blend_mask)).astype(np.uint8)

        # Put back into frame
        output = frame.copy()
        output[y1_exp:y2_exp, x1_exp:x2_exp] = blended_face

        return output

    except Exception as e:
        print(f"Warning: SR3 enhancement failed: {e}")
        return frame


def get_diffusion_models_info():
    """Return information about available diffusion models."""
    info = []
    for name, config in DIFFUSION_MODELS.items():
        model_type = config.get('type', 'unknown')
        if 'size' in config:
            info.append(f"  {name}: {config['description']} (size: {config['size']}, type: {model_type})")
        else:
            info.append(f"  {name}: {config['description']} (type: {model_type})")
    return "\n".join(info)
