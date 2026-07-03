from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from bogvm.receipts import sha256_hex


class CandidateType(str, Enum):
    VERIFIED_MAXIMUM = "VERIFIED_MAXIMUM"
    VERIFIED_MINIMUM = "VERIFIED_MINIMUM"
    VERIFIED_SADDLE = "VERIFIED_SADDLE"
    DEGENERATE = "DEGENERATE"
    UNRESOLVED = "UNRESOLVED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class VerificationReceipt:
    grid_index: tuple[int, int]
    gradient_magnitude: float
    eigenvalues: tuple[float, float]
    classification: CandidateType

    def to_json(self) -> dict[str, Any]:
        return {
            "grid_index": list(self.grid_index),
            "gradient_magnitude": self.gradient_magnitude,
            "eigenvalues": list(self.eigenvalues),
            "classification": self.classification.value,
        }


class TopologyCompiler:
    def __init__(self, gradient_tolerance: float = 1e-6, curvature_tolerance: float = 1e-6):
        self.gradient_tolerance = gradient_tolerance
        self.curvature_tolerance = curvature_tolerance

    def diagnostics(self, phi: np.ndarray, spacing: float = 1.0) -> dict[str, np.ndarray]:
        gx, gy = np.gradient(phi, spacing, spacing, edge_order=2)
        gxx, gxy = np.gradient(gx, spacing, spacing, edge_order=2)
        gyx, gyy = np.gradient(gy, spacing, spacing, edge_order=2)
        hxy = 0.5 * (gxy + gyx)
        eigs = np.empty(phi.shape + (2,), dtype=np.float64)
        for idx in np.ndindex(phi.shape):
            eigs[idx] = np.linalg.eigvalsh([[gxx[idx], hxy[idx]], [hxy[idx], gyy[idx]]])
        return {"grad_x": gx, "grad_y": gy, "grad_mag": np.sqrt(gx * gx + gy * gy), "lambda_1": eigs[..., 0], "lambda_2": eigs[..., 1]}

    def classify(self, diagnostics: dict[str, np.ndarray], idx: tuple[int, int]) -> VerificationReceipt:
        g = float(diagnostics["grad_mag"][idx])
        l1 = float(diagnostics["lambda_1"][idx])
        l2 = float(diagnostics["lambda_2"][idx])
        tol = self.curvature_tolerance
        if g > self.gradient_tolerance:
            kind = CandidateType.REJECTED
        elif abs(l1) <= tol or abs(l2) <= tol:
            kind = CandidateType.DEGENERATE
        elif l1 < -tol and l2 < -tol:
            kind = CandidateType.VERIFIED_MAXIMUM
        elif l1 > tol and l2 > tol:
            kind = CandidateType.VERIFIED_MINIMUM
        elif l1 < -tol and l2 > tol:
            kind = CandidateType.VERIFIED_SADDLE
        else:
            kind = CandidateType.UNRESOLVED
        return VerificationReceipt(idx, g, (l1, l2), kind)

    def compile_graph(self, phi: np.ndarray, spacing: float = 1.0) -> dict[str, Any]:
        diagnostics = self.diagnostics(phi, spacing)
        nodes: list[dict[str, Any]] = []
        saddles: list[dict[str, Any]] = []
        receipts: list[dict[str, Any]] = []
        for idx in np.ndindex(phi.shape):
            receipt = self.classify(diagnostics, idx)
            if receipt.classification in {
                CandidateType.VERIFIED_MAXIMUM,
                CandidateType.VERIFIED_MINIMUM,
                CandidateType.VERIFIED_SADDLE,
                CandidateType.DEGENERATE,
            }:
                receipts.append(receipt.to_json())
            if receipt.classification == CandidateType.VERIFIED_MAXIMUM:
                nodes.append({"id": len(nodes), "grid_index": list(idx), "verification": receipt.to_json()})
            elif receipt.classification == CandidateType.VERIFIED_SADDLE:
                saddles.append({"id": len(saddles), "grid_index": list(idx), "verification": receipt.to_json()})
        graph = {
            "format": "boggraph",
            "schema_version": 0,
            "compiler": "tstopology-v0",
            "compiler_configuration": {
                "gradient_tolerance": self.gradient_tolerance,
                "curvature_tolerance": self.curvature_tolerance,
                "spacing": spacing,
            },
            "nodes": nodes,
            "saddles": saddles,
            "edges": [],
            "compiler_receipts": receipts,
        }
        graph["graph_hash"] = sha256_hex(graph)
        return graph
