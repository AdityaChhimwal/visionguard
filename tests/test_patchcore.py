import numpy as np
import pytest

pytestmark = pytest.mark.ml


def test_real_patchcore_coreset_and_graph_parity(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("anomalib")
    from visionguard.ml.artifacts import Graph
    from visionguard.ml.patchcore import ATOL, RTOL, ExportOutputs, build, export, fit

    torch.set_num_threads(1)
    torch.manual_seed(7)
    config = {"backbone": "resnet18.a1_in1k", "layers": ["layer2", "layer3"], "neighbors": 1}
    rng = np.random.default_rng(7)
    fitting = rng.standard_normal((5, 3, 64, 64), dtype=np.float32)
    query = rng.standard_normal((1, 3, 64, 64), dtype=np.float32)
    model = build(config, None)
    counts = fit(model, [fitting], 0.1)
    assert counts == {"fit_patches": 320, "coreset_rows": 32, "embedding_channels": 384}
    path = tmp_path / "model.onnx"
    export(model, query, path)
    graph = Graph(path.read_bytes(), 64, 1)
    with torch.inference_mode():
        ref_score, ref_map = ExportOutputs(model)(torch.from_numpy(query))
    score, heatmap = graph.run(query)
    assert heatmap.shape == (64, 64) and np.isfinite(heatmap).all() and np.isfinite(score)
    assert score == pytest.approx(float(ref_score.item()), abs=ATOL, rel=RTOL)
    np.testing.assert_allclose(heatmap, ref_map.numpy()[0, 0], rtol=RTOL, atol=ATOL)
    assert Graph(path.read_bytes(), 64, 1).run(query)[0] == score
