import numpy as np

from tstopology import CandidateType, TopologyCompiler


def test_maximum_and_saddle_signatures_are_separate():
    compiler = TopologyCompiler(gradient_tolerance=1e-9, curvature_tolerance=1e-9)
    x = np.linspace(-1, 1, 9)
    X, Y = np.meshgrid(x, x, indexing="ij")
    max_phi = -(X**2 + Y**2)
    max_diag = compiler.diagnostics(max_phi, x[1] - x[0])
    assert compiler.classify(max_diag, (4, 4)).classification == CandidateType.VERIFIED_MAXIMUM
    saddle_phi = -(X**2) + Y**2
    saddle_diag = compiler.diagnostics(saddle_phi, x[1] - x[0])
    assert compiler.classify(saddle_diag, (4, 4)).classification == CandidateType.VERIFIED_SADDLE


def test_rejected_candidates_do_not_enter_graph_and_hash_stable():
    compiler = TopologyCompiler(gradient_tolerance=1e-9, curvature_tolerance=1e-9)
    x = np.linspace(-1, 1, 9)
    X, Y = np.meshgrid(x, x, indexing="ij")
    graph1 = compiler.compile_graph(X + Y, x[1] - x[0])
    graph2 = compiler.compile_graph(X + Y, x[1] - x[0])
    assert graph1["nodes"] == []
    assert graph1["saddles"] == []
    assert graph1["graph_hash"] == graph2["graph_hash"]
    assert graph1["compiler_configuration"]["spacing"] == x[1] - x[0]
