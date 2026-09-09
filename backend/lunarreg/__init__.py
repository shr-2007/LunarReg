"""LunarReg: geometry-aware lunar image registration."""

from .config import RegistrationConfig, load_config
from .registration import RegistrationError, register_files, register_images

__all__ = [
    "RegistrationConfig",
    "RegistrationError",
    "load_config",
    "register_files",
    "register_images",
]

__version__ = "0.1.0"
