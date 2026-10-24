"""Every test is offline, including accidentally invoked real model loaders."""

import socket

import pytest


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Network/Hub access or pretrained weight loading is forbidden in tests"
        )

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    import huggingface_hub
    import transformers
    from sentence_transformers import SentenceTransformer

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", forbidden)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", forbidden)
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", forbidden)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", forbidden)
    monkeypatch.setattr(transformers.PreTrainedModel, "from_pretrained", forbidden)
    monkeypatch.setattr(SentenceTransformer, "__init__", forbidden)
