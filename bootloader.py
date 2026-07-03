"""Compatibility wrapper for canonical .bogpkg boot loading.

The previous .bogpk loader is preserved under
legacy/experiments/original_root/bootloader.py.
"""

from tsos.bootloader import boot_package

load_package = boot_package

__all__ = ["boot_package", "load_package"]
