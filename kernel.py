"""Compatibility wrapper for the canonical TS-OS kernel.

The previous prototype implementation is preserved under
legacy/experiments/original_root/kernel.py.
"""

from tsos.kernel import IPCRecord, Kernel, Process, ProcessStatus, UNOWNED, WeightedSpatialAllocator

__all__ = ["IPCRecord", "Kernel", "Process", "ProcessStatus", "UNOWNED", "WeightedSpatialAllocator"]
