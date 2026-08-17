import numpy as np

from r16p19.phase5_verifier_model import LinearVerifier, auroc, choose_threshold, metrics


def test_metrics_and_linear_fit_have_expected_direction():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(300, 4))
    y = (x[:, 0] + x[:, 1] > 0).astype(np.float32)
    model = LinearVerifier().fit(x, y, steps=300)
    scores = model.predict(x)
    threshold = choose_threshold(y, scores)
    assert auroc(y, scores) > 0.95
    assert metrics(y, scores, threshold)["accuracy"] > 0.85
