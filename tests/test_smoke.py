"""
Smoke tests validate imports and basic utilities
"""

import importlib


def test_stdlib_imports():
    """Core stdlib modules"""
    for mod in ["os", "re", "logging", "argparse", "time"]:
        assert importlib.import_module(mod) is not None


def test_numpy_importable():
    import numpy as np

    arr = np.array([1, 2, 3])
    assert arr.sum() == 6


def test_pandas_importable():
    import pandas as pd

    df = pd.DataFrame({"label": [0, 1], "text": ["real", "fake"]})
    assert len(df) == 2
    assert list(df.columns) == ["label", "text"]


def test_sklearn_preprocessing():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split

    texts = ["this is real news", "this is fake news", "another real article", "more fake"]
    labels = [0, 1, 0, 1]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    assert X.shape[0] == 4

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.5, random_state=42)
    assert X_train.shape[0] == 2
    assert X_test.shape[0] == 2


def test_torch_importable():
    import torch

    t = torch.tensor([1.0, 2.0, 3.0])
    assert t.shape == (3,)
    assert t.sum().item() == 6.0


def test_torch_nn_linear():
    import torch
    import torch.nn as nn

    layer = nn.Linear(10, 2)
    x = torch.randn(4, 10)
    out = layer(x)
    assert out.shape == (4, 2)


def test_transformers_tokenizer_loads():
    """Verify the transformers library"""
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        tokens = tokenizer("hello world", return_tensors="pt")
        assert "input_ids" in tokens
    except OSError:
        # Offline / no HuggingFace cache — skip rather than fail
        import pytest

        pytest.skip("HuggingFace model not cached; skipping tokenizer test in offline CI")


def test_metrics_functions():
    from sklearn.metrics import accuracy_score, f1_score

    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    assert acc == 0.75
    assert round(f1, 4) == round(2 * 2 / (2 + 2 + 1), 4)  # TP=2, FP=1, FN=0 - 0.8
