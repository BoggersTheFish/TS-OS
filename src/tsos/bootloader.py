from __future__ import annotations

from pathlib import Path

from bogvm.formats import BogPkg, read_json
from .kernel import Kernel


def boot_package(path: str | Path, kernel: Kernel | None = None) -> Kernel:
    package = BogPkg.from_json(read_json(path))
    kernel = kernel or Kernel()
    kernel.boot_package(package)
    return kernel
