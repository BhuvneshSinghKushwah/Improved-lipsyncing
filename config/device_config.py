"""
Device configuration and Apple Silicon detection.

Detects M1/M2/M3/M4 chip variants and their capabilities,
including Neural Engine availability and optimal batch sizes.
"""

import platform
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import os


class ChipGeneration(Enum):
    """Apple Silicon chip generations."""
    UNKNOWN = "unknown"
    M1 = "m1"
    M2 = "m2"
    M3 = "m3"
    M4 = "m4"
    INTEL = "intel"


class ChipVariant(Enum):
    """Apple Silicon chip variants."""
    BASE = "base"
    PRO = "pro"
    MAX = "max"
    ULTRA = "ultra"


@dataclass
class DeviceInfo:
    """Information about the current device."""
    chip_generation: ChipGeneration
    chip_variant: ChipVariant
    chip_name: str
    cpu_cores: int
    gpu_cores: int
    neural_engine_cores: int
    unified_memory_gb: float
    has_mps: bool
    has_neural_engine: bool

    # Recommended batch sizes based on device capabilities
    recommended_wav2lip_batch_size: int
    recommended_face_det_batch_size: int
    recommended_training_batch_size: int

    def __str__(self) -> str:
        return (
            f"DeviceInfo(\n"
            f"  chip: {self.chip_name}\n"
            f"  generation: {self.chip_generation.value}\n"
            f"  variant: {self.chip_variant.value}\n"
            f"  cpu_cores: {self.cpu_cores}\n"
            f"  gpu_cores: {self.gpu_cores}\n"
            f"  neural_engine_cores: {self.neural_engine_cores}\n"
            f"  unified_memory: {self.unified_memory_gb}GB\n"
            f"  has_mps: {self.has_mps}\n"
            f"  has_neural_engine: {self.has_neural_engine}\n"
            f")"
        )


def _get_sysctl_value(key: str) -> Optional[str]:
    """Get a sysctl value."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", key],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def _get_system_profiler_info() -> dict:
    """Get hardware info from system_profiler."""
    info = {}
    try:
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            if "SPHardwareDataType" in data and len(data["SPHardwareDataType"]) > 0:
                hw = data["SPHardwareDataType"][0]
                info["chip_type"] = hw.get("chip_type", "")
                info["machine_model"] = hw.get("machine_model", "")
                info["physical_memory"] = hw.get("physical_memory", "")
                info["number_processors"] = hw.get("number_processors", "")
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    return info


def _parse_chip_info(chip_type: str) -> tuple[ChipGeneration, ChipVariant]:
    """Parse chip type string to generation and variant."""
    chip_lower = chip_type.lower()

    # Determine generation
    generation = ChipGeneration.UNKNOWN
    if "m4" in chip_lower:
        generation = ChipGeneration.M4
    elif "m3" in chip_lower:
        generation = ChipGeneration.M3
    elif "m2" in chip_lower:
        generation = ChipGeneration.M2
    elif "m1" in chip_lower:
        generation = ChipGeneration.M1
    elif "intel" in chip_lower:
        generation = ChipGeneration.INTEL

    # Determine variant
    variant = ChipVariant.BASE
    if "ultra" in chip_lower:
        variant = ChipVariant.ULTRA
    elif "max" in chip_lower:
        variant = ChipVariant.MAX
    elif "pro" in chip_lower:
        variant = ChipVariant.PRO

    return generation, variant


def _get_gpu_cores(generation: ChipGeneration, variant: ChipVariant) -> int:
    """Estimate GPU cores based on chip generation and variant."""
    # Approximate GPU core counts for Apple Silicon
    gpu_cores = {
        (ChipGeneration.M1, ChipVariant.BASE): 8,
        (ChipGeneration.M1, ChipVariant.PRO): 16,
        (ChipGeneration.M1, ChipVariant.MAX): 32,
        (ChipGeneration.M1, ChipVariant.ULTRA): 64,
        (ChipGeneration.M2, ChipVariant.BASE): 10,
        (ChipGeneration.M2, ChipVariant.PRO): 19,
        (ChipGeneration.M2, ChipVariant.MAX): 38,
        (ChipGeneration.M2, ChipVariant.ULTRA): 76,
        (ChipGeneration.M3, ChipVariant.BASE): 10,
        (ChipGeneration.M3, ChipVariant.PRO): 18,
        (ChipGeneration.M3, ChipVariant.MAX): 40,
        (ChipGeneration.M4, ChipVariant.BASE): 10,
        (ChipGeneration.M4, ChipVariant.PRO): 20,
        (ChipGeneration.M4, ChipVariant.MAX): 40,
    }
    return gpu_cores.get((generation, variant), 8)


def _get_neural_engine_cores(generation: ChipGeneration) -> int:
    """Get Neural Engine core count based on chip generation."""
    ne_cores = {
        ChipGeneration.M1: 16,
        ChipGeneration.M2: 16,
        ChipGeneration.M3: 16,
        ChipGeneration.M4: 16,  # M4 has enhanced 16-core ANE
        ChipGeneration.INTEL: 0,
        ChipGeneration.UNKNOWN: 0,
    }
    return ne_cores.get(generation, 0)


def _get_recommended_batch_sizes(
    generation: ChipGeneration,
    variant: ChipVariant,
    memory_gb: float
) -> tuple[int, int, int]:
    """Get recommended batch sizes based on device capabilities."""
    # Base recommendations scaled by memory
    if memory_gb >= 64:
        wav2lip_batch = 128
        face_det_batch = 16
        training_batch = 32
    elif memory_gb >= 32:
        wav2lip_batch = 64
        face_det_batch = 8
        training_batch = 16
    elif memory_gb >= 16:
        wav2lip_batch = 32
        face_det_batch = 4
        training_batch = 8
    else:
        wav2lip_batch = 16
        face_det_batch = 2
        training_batch = 4

    # Adjust for newer generations (M3/M4 have better efficiency)
    if generation in [ChipGeneration.M3, ChipGeneration.M4]:
        wav2lip_batch = int(wav2lip_batch * 1.25)
        training_batch = int(training_batch * 1.25)

    # Cap at reasonable limits
    wav2lip_batch = min(wav2lip_batch, 256)
    face_det_batch = min(face_det_batch, 32)
    training_batch = min(training_batch, 64)

    return wav2lip_batch, face_det_batch, training_batch


def _check_mps_available() -> bool:
    """Check if MPS (Metal Performance Shaders) is available."""
    try:
        import torch
        return torch.backends.mps.is_available() and torch.backends.mps.is_built()
    except ImportError:
        return False


def _check_mlx_available() -> bool:
    """Check if MLX is available."""
    try:
        import mlx.core as mx
        # Simple test to verify MLX works
        _ = mx.array([1, 2, 3])
        return True
    except ImportError:
        return False


def _check_coreml_available() -> bool:
    """Check if CoreML tools are available."""
    try:
        import coremltools
        return True
    except ImportError:
        return False


def get_device_info() -> DeviceInfo:
    """
    Detect and return information about the current device.

    Returns:
        DeviceInfo object with device capabilities and recommendations.
    """
    # Check if running on macOS
    if platform.system() != "Darwin":
        return DeviceInfo(
            chip_generation=ChipGeneration.UNKNOWN,
            chip_variant=ChipVariant.BASE,
            chip_name="Non-macOS System",
            cpu_cores=os.cpu_count() or 4,
            gpu_cores=0,
            neural_engine_cores=0,
            unified_memory_gb=8.0,
            has_mps=False,
            has_neural_engine=False,
            recommended_wav2lip_batch_size=16,
            recommended_face_det_batch_size=4,
            recommended_training_batch_size=4,
        )

    # Get system info
    sys_info = _get_system_profiler_info()
    chip_type = sys_info.get("chip_type", "")

    # Parse chip generation and variant
    generation, variant = _parse_chip_info(chip_type)

    # Get CPU cores
    cpu_cores_str = _get_sysctl_value("hw.ncpu")
    cpu_cores = int(cpu_cores_str) if cpu_cores_str else (os.cpu_count() or 4)

    # Get memory
    memory_str = _get_sysctl_value("hw.memsize")
    if memory_str:
        memory_gb = int(memory_str) / (1024**3)
    else:
        memory_gb = 8.0

    # Get GPU and Neural Engine cores
    gpu_cores = _get_gpu_cores(generation, variant)
    ne_cores = _get_neural_engine_cores(generation)

    # Check capabilities
    has_mps = _check_mps_available()
    has_neural_engine = ne_cores > 0 and generation != ChipGeneration.INTEL

    # Get recommended batch sizes
    wav2lip_batch, face_det_batch, training_batch = _get_recommended_batch_sizes(
        generation, variant, memory_gb
    )

    return DeviceInfo(
        chip_generation=generation,
        chip_variant=variant,
        chip_name=chip_type if chip_type else f"{platform.processor()} (detected)",
        cpu_cores=cpu_cores,
        gpu_cores=gpu_cores,
        neural_engine_cores=ne_cores,
        unified_memory_gb=round(memory_gb, 1),
        has_mps=has_mps,
        has_neural_engine=has_neural_engine,
        recommended_wav2lip_batch_size=wav2lip_batch,
        recommended_face_det_batch_size=face_det_batch,
        recommended_training_batch_size=training_batch,
    )


def get_optimal_device() -> str:
    """
    Get the optimal device string for PyTorch.

    Returns:
        "mps" if available, otherwise "cpu".
    """
    if _check_mps_available():
        return "mps"
    return "cpu"


def print_device_info():
    """Print device information to console."""
    info = get_device_info()
    print(info)
    print(f"\nFramework Availability:")
    print(f"  PyTorch MPS: {_check_mps_available()}")
    print(f"  MLX: {_check_mlx_available()}")
    print(f"  CoreML Tools: {_check_coreml_available()}")


if __name__ == "__main__":
    print_device_info()
