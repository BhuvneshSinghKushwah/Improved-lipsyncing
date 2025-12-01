from os import listdir, path
import gc
import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
from models import Wav2Lip
import platform

# GFPGAN imports (optional, for face enhancement)
GFPGAN_AVAILABLE = False
try:
	from gfpgan import GFPGANer
	GFPGAN_AVAILABLE = True
except ImportError:
	pass

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str, 
					help='Name of saved checkpoint to load weights from', required=True)

parser.add_argument('--face', type=str, 
					help='Filepath of video/image that contains faces to use', required=True)
parser.add_argument('--audio', type=str, 
					help='Filepath of video/audio file to use as raw audio source', required=True)
parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.', 
								default='results/result_voice.mp4')

parser.add_argument('--static', type=bool, 
					help='If True, then use only first video frame for inference', default=False)
parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', 
					default=25., required=False)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
					help='Padding (top, bottom, left, right). Please adjust to include chin at least')

parser.add_argument('--face_det_batch_size', type=int,
					help='Batch size for face detection', default=4)
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)

parser.add_argument('--resize_factor', default=1, type=int, 
			help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], 
					help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. ' 
					'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], 
					help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
					'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

parser.add_argument('--rotate', default=False, action='store_true',
					help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
					'Use if you get a flipped result, despite feeding a normal looking video')

parser.add_argument('--nosmooth', default=False, action='store_true',
					help='Prevent smoothing face detections over a short temporal window')

parser.add_argument('--no_feather', default=False, action='store_true',
					help='Disable feathered blending for mouth region (use hard paste instead)')

parser.add_argument('--feather_amount', type=int, default=3,
					help='Amount of feathering for mouth blend (default: 3)')

parser.add_argument('--enhance', default=False, action='store_true',
					help='Apply GFPGAN face enhancement after Wav2Lip')

parser.add_argument('--enhancer_model', type=str, default='GFPGANv1.4',
					help='GFPGAN model version: GFPGANv1.3, GFPGANv1.4 (default: GFPGANv1.4)')

parser.add_argument('--enhance_interval', type=int, default=1,
					help='Apply GFPGAN enhancement every N frames (default: 1 = every frame). Higher values = faster but lower quality.')

parser.add_argument('--enhance_blend', type=float, default=0.0,
					help='Temporal blending for GFPGAN (0.0-0.9). Higher = smoother but more blur. Try 0.3-0.5 to reduce flicker.')

parser.add_argument('--face_det_interval', type=int, default=1,
					help='Run face detection every N frames and interpolate (default: 1 = every frame). Higher values = much faster face detection.')

args = parser.parse_args()
args.img_size = 96

if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
	args.static = True

def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

def create_feathered_mask(height, width, feather_amount=3):
	"""Create a feathered mask for smooth blending of the mouth region."""
	mask = np.ones((height, width), dtype=np.float32)

	# Create feathered edges
	feather = feather_amount
	for i in range(feather):
		alpha = (i + 1) / (feather + 1)
		# Top edge
		mask[i, :] = alpha
		# Bottom edge
		mask[height - 1 - i, :] = alpha
		# Left edge
		mask[:, i] = np.minimum(mask[:, i], alpha)
		# Right edge
		mask[:, width - 1 - i] = np.minimum(mask[:, width - 1 - i], alpha)

	# Expand to 3 channels
	mask = np.stack([mask] * 3, axis=-1)
	return mask

def blend_with_feather(frame, pred, coords, feather_amount=3):
	"""Blend predicted face region into frame with feathered edges."""
	y1, y2, x1, x2 = coords
	h, w = y2 - y1, x2 - x1

	# Resize prediction to target size
	pred_resized = cv2.resize(pred.astype(np.uint8), (w, h))

	# Create feathered mask
	mask = create_feathered_mask(h, w, feather_amount)

	# Get the original region
	original_region = frame[y1:y2, x1:x2].astype(np.float32)
	pred_float = pred_resized.astype(np.float32)

	# Blend using the mask
	blended = (pred_float * mask + original_region * (1 - mask)).astype(np.uint8)

	# Place back into frame
	frame[y1:y2, x1:x2] = blended
	return frame

def load_gfpgan_enhancer(model_name='GFPGANv1.4', device='cuda'):
	"""Load GFPGAN face enhancer model."""
	if not GFPGAN_AVAILABLE:
		raise ImportError("GFPGAN not installed. Run: pip install gfpgan")

	# Model paths
	model_paths = {
		'GFPGANv1.3': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
		'GFPGANv1.4': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth',
	}

	model_path = model_paths.get(model_name, model_paths['GFPGANv1.4'])

	# Initialize GFPGAN
	restorer = GFPGANer(
		model_path=model_path,
		upscale=1,
		arch='clean',
		channel_multiplier=2,
		bg_upsampler=None,
		device=device
	)

	return restorer

def enhance_frame_gfpgan(frame, restorer):
	"""Enhance a single frame using GFPGAN."""
	# GFPGAN expects BGR input
	_, _, output = restorer.enhance(
		frame,
		has_aligned=False,
		only_center_face=True,
		paste_back=True
	)
	return output

def interpolate_boxes(boxes, total_frames, detected_indices):
	"""Interpolate bounding boxes for frames that weren't detected."""
	all_boxes = np.zeros((total_frames, 4))

	for i in range(total_frames):
		if i in detected_indices:
			# Use the detected box
			det_idx = detected_indices.index(i)
			all_boxes[i] = boxes[det_idx]
		else:
			# Find surrounding detected frames and interpolate
			prev_idx = None
			next_idx = None
			for j in detected_indices:
				if j < i:
					prev_idx = j
				elif j > i and next_idx is None:
					next_idx = j
					break

			if prev_idx is not None and next_idx is not None:
				# Linear interpolation
				prev_det = detected_indices.index(prev_idx)
				next_det = detected_indices.index(next_idx)
				t = (i - prev_idx) / (next_idx - prev_idx)
				all_boxes[i] = boxes[prev_det] * (1 - t) + boxes[next_det] * t
			elif prev_idx is not None:
				# Use previous
				all_boxes[i] = boxes[detected_indices.index(prev_idx)]
			elif next_idx is not None:
				# Use next
				all_boxes[i] = boxes[detected_indices.index(next_idx)]

	return all_boxes

def face_detect(images, interval=1):
	"""
	Detect faces in images.

	Args:
		images: List of images to detect faces in
		interval: Detect every Nth frame and interpolate (default: 1 = every frame)
	"""
	detector = get_face_detector()
	batch_size = args.face_det_batch_size

	# Select frames for detection based on interval
	if interval > 1:
		detect_indices = list(range(0, len(images), interval))
		# Always include the last frame for better interpolation
		if detect_indices[-1] != len(images) - 1:
			detect_indices.append(len(images) - 1)
		images_to_detect = [images[i] for i in detect_indices]
		print(f"Face detection: processing {len(images_to_detect)}/{len(images)} frames (interval={interval})")
	else:
		detect_indices = list(range(len(images)))
		images_to_detect = images

	while 1:
		predictions = []
		try:
			for i in tqdm(range(0, len(images_to_detect), batch_size), desc="Face detection"):
				batch_imgs = np.array(images_to_detect[i:i + batch_size])
				batch_preds = detector.get_detections_for_batch(batch_imgs)
				predictions.extend(batch_preds)
				del batch_imgs  # Free memory immediately
				# Clear memory after each batch to prevent buildup on MPS
				gc.collect()
				if device == 'mps':
					torch.mps.empty_cache()
				elif device == 'cuda':
					torch.cuda.empty_cache()
		except RuntimeError as e:
			if batch_size == 1:
				raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
			batch_size //= 2
			print('Recovering from OOM error; New batch size: {}'.format(batch_size))
			# Clear memory before retry
			if device == 'mps':
				torch.mps.empty_cache()
			elif device == 'cuda':
				torch.cuda.empty_cache()
			continue
		break

	# Process detected frames
	detected_boxes = []
	pady1, pady2, padx1, padx2 = args.pads
	failed_frames = []

	for idx, (rect, image) in enumerate(zip(predictions, images_to_detect)):
		if rect is None:
			failed_frames.append(detect_indices[idx])
			# Use fallback
			if detected_boxes:
				detected_boxes.append(detected_boxes[-1].copy())
			else:
				cv2.imwrite('temp/faulty_frame.jpg', image)
				raise ValueError(f'Face not detected in frame {detect_indices[idx]}! Ensure the video contains a face in all the frames.')
		else:
			y1 = max(0, rect[1] - pady1)
			y2 = min(image.shape[0], rect[3] + pady2)
			x1 = max(0, rect[0] - padx1)
			x2 = min(image.shape[1], rect[2] + padx2)
			detected_boxes.append([x1, y1, x2, y2])

	if failed_frames:
		print(f"Warning: Face not detected in {len(failed_frames)} frames, using interpolated detections")

	# Interpolate boxes for all frames if using interval
	if interval > 1:
		boxes = interpolate_boxes(np.array(detected_boxes), len(images), detect_indices)
	else:
		boxes = np.array(detected_boxes)

	if not args.nosmooth:
		boxes = get_smoothened_boxes(boxes, T=5)

	results = [[image[int(y1):int(y2), int(x1):int(x2)], (int(y1), int(y2), int(x1), int(x2))]
			   for image, (x1, y1, x2, y2) in zip(images, boxes)]

	return results 

def datagen(frames, mels, face_det_results=None):
	"""
	Generator that yields batches of images, mel spectrograms, frames, and coordinates.

	Args:
		frames: List of video frames
		mels: List of mel spectrogram chunks
		face_det_results: Pre-computed face detection results (optional, will compute if None)
	"""
	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	# Use pre-computed face detection results if provided
	if face_det_results is None:
		if args.box[0] == -1:
			if not args.static:
				face_det_results = face_detect(frames)
			else:
				face_det_results = face_detect([frames[0]])
		else:
			print('Using the specified bounding box instead of face detection...')
			y1, y2, x1, x2 = args.box
			face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

	for i, m in enumerate(mels):
		idx = 0 if args.static else i%len(frames)
		frame_to_save = frames[idx].copy()
		face, coords = face_det_results[idx].copy()

		face = cv2.resize(face, (args.img_size, args.img_size))
			
		img_batch.append(face)
		mel_batch.append(m)
		frame_batch.append(frame_to_save)
		coords_batch.append(coords)

		if len(img_batch) >= args.wav2lip_batch_size:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

			img_masked = img_batch.copy()
			img_masked[:, args.img_size//2:] = 0

			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			yield img_batch, mel_batch, frame_batch, coords_batch
			img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if len(img_batch) > 0:
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, args.img_size//2:] = 0

		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		yield img_batch, mel_batch, frame_batch, coords_batch

mel_step_size = 16
if torch.cuda.is_available():
	device = 'cuda'
elif torch.backends.mps.is_available():
	device = 'mps'
else:
	device = 'cpu'
print('Using {} for inference.'.format(device))

# Global face detector (created once, reused)
_face_detector = None

def get_face_detector():
	"""Get or create the global face detector (singleton pattern)."""
	global _face_detector
	if _face_detector is None:
		_face_detector = face_detection.FaceAlignment(
			face_detection.LandmarksType._2D,
			flip_input=False,
			device=device
		)
	return _face_detector

def cleanup_face_detector():
	"""Clean up the face detector to free memory."""
	global _face_detector
	if _face_detector is not None:
		del _face_detector
		_face_detector = None
		if device == 'cuda':
			torch.cuda.empty_cache()
		elif device == 'mps':
			torch.mps.empty_cache()

def _load(checkpoint_path):
	if device == 'cuda':
		checkpoint = torch.load(checkpoint_path, weights_only=False)
	else:
		checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
	return checkpoint

def load_model(path):
	print("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)

	# Check if it's a TorchScript model
	if isinstance(checkpoint, torch.jit.ScriptModule):
		model = checkpoint.to(device)
		return model.eval()

	# Regular checkpoint with state_dict
	model = Wav2Lip()
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	model = model.to(device)
	return model.eval()

def main():
	if not os.path.isfile(args.face):
		raise ValueError('--face argument must be a valid path to video/image file')

	elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
		full_frames = [cv2.imread(args.face)]
		fps = args.fps

	else:
		video_stream = cv2.VideoCapture(args.face)
		fps = video_stream.get(cv2.CAP_PROP_FPS)

		print('Reading video frames...')

		full_frames = []
		while 1:
			still_reading, frame = video_stream.read()
			if not still_reading:
				video_stream.release()
				break
			if args.resize_factor > 1:
				frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))

			if args.rotate:
				frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

			y1, y2, x1, x2 = args.crop
			if x2 == -1: x2 = frame.shape[1]
			if y2 == -1: y2 = frame.shape[0]

			frame = frame[y1:y2, x1:x2]

			full_frames.append(frame)

	print ("Number of frames available for inference: "+str(len(full_frames)))

	if not args.audio.endswith('.wav'):
		print('Extracting raw audio...')
		command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav')

		subprocess.call(command, shell=True)
		args.audio = 'temp/temp.wav'

	wav = audio.load_wav(args.audio, 16000)
	mel = audio.melspectrogram(wav)
	print(mel.shape)

	if np.isnan(mel.reshape(-1)).sum() > 0:
		raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

	mel_chunks = []
	mel_idx_multiplier = 80./fps 
	i = 0
	while 1:
		start_idx = int(i * mel_idx_multiplier)
		if start_idx + mel_step_size > len(mel[0]):
			mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
			break
		mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
		i += 1

	print("Length of mel chunks: {}".format(len(mel_chunks)))

	full_frames = full_frames[:len(mel_chunks)]

	batch_size = args.wav2lip_batch_size

	# Run face detection FIRST (before loading GFPGAN to avoid memory issues)
	print("Running face detection...")
	if args.box[0] == -1:
		if not args.static:
			face_det_results = face_detect(full_frames, interval=args.face_det_interval)
		else:
			face_det_results = face_detect([full_frames[0]], interval=1)
	else:
		print('Using the specified bounding box instead of face detection...')
		y1, y2, x1, x2 = args.box
		face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in full_frames]
	print(f"Face detection complete: {len(face_det_results)} face regions extracted")

	# Clean up face detector to free memory before loading GFPGAN
	cleanup_face_detector()
	if device == 'mps':
		torch.mps.empty_cache()
	elif device == 'cuda':
		torch.cuda.empty_cache()

	# Now create the generator with pre-computed face detection results
	gen = datagen(full_frames.copy(), mel_chunks, face_det_results=face_det_results)

	# Load GFPGAN enhancer if requested (after face detection is done)
	gfpgan_restorer = None
	if args.enhance:
		if not GFPGAN_AVAILABLE:
			print("WARNING: GFPGAN not installed. Run: pip install gfpgan")
			print("Continuing without enhancement...")
		else:
			print("Loading GFPGAN enhancer...")
			gfpgan_restorer = load_gfpgan_enhancer(args.enhancer_model, device)
			print("GFPGAN loaded successfully")

	# Determine blending mode
	use_feather = not args.no_feather
	if use_feather:
		print(f"Using feathered blending (amount: {args.feather_amount})")
	else:
		print("Using hard paste (no feathering)")

	if gfpgan_restorer is not None and args.enhance_interval > 1:
		print(f"GFPGAN enhancement: every {args.enhance_interval} frames (faster mode)")
	if gfpgan_restorer is not None and args.enhance_blend > 0:
		print(f"GFPGAN temporal blending: {args.enhance_blend} (reduces flicker)")

	frame_count = 0  # Global frame counter for enhance_interval
	enhanced_count = 0  # Count of enhanced frames
	prev_enhanced_frame = None  # For temporal blending

	for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen,
											total=int(np.ceil(float(len(mel_chunks))/batch_size)),
											desc="Wav2Lip inference")):
		if i == 0:
			model = load_model(args.checkpoint_path)
			print("Model loaded")

			frame_h, frame_w = full_frames[0].shape[:-1]
			out = cv2.VideoWriter('temp/result.avi',
									cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

		img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
		mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

		with torch.no_grad():
			pred = model(mel_batch, img_batch)

		pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

		for p, f, c in zip(pred, frames, coords):
			# Apply feathered blending or hard paste
			if use_feather:
				f = blend_with_feather(f, p, c, args.feather_amount)
			else:
				y1, y2, x1, x2 = c
				p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
				f[y1:y2, x1:x2] = p

			# Apply GFPGAN enhancement if enabled (respecting enhance_interval)
			if gfpgan_restorer is not None and (frame_count % args.enhance_interval == 0):
				try:
					enhanced = enhance_frame_gfpgan(f, gfpgan_restorer)

					# Apply temporal blending to reduce flicker
					if args.enhance_blend > 0 and prev_enhanced_frame is not None:
						blend_factor = args.enhance_blend
						enhanced = cv2.addWeighted(
							enhanced, 1.0 - blend_factor,
							prev_enhanced_frame, blend_factor,
							0
						)

					prev_enhanced_frame = enhanced.copy()
					f = enhanced
					enhanced_count += 1
				except Exception:
					pass  # Skip enhancement on error, use original frame

			frame_count += 1
			out.write(f)

	if gfpgan_restorer is not None:
		print(f"Enhanced {enhanced_count}/{frame_count} frames with GFPGAN")

	out.release()

	command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, 'temp/result.avi', args.outfile)
	subprocess.call(command, shell=platform.system() != 'Windows')

if __name__ == '__main__':
	main()
