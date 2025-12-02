"""Configuration module for macOS-optimized lipsyncing."""

from .device_config import (
    DeviceInfo,
    ChipGeneration,
    ChipVariant,
    get_device_info,
    get_optimal_device,
    print_device_info,
)

__all__ = [
    "DeviceInfo",
    "ChipGeneration",
    "ChipVariant",
    "get_device_info",
    "get_optimal_device",
    "print_device_info",
]
