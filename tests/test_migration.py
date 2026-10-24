import importlib
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import yaml
from PIL import Image

from image_retrieval.config import ConfigError, Profile, Runtime, load_settings
from image_retrieval.features import FeatureStore, StoreError, writer_lock
from image_retrieval.images import load_image
from image_retrieval.indexing import encode_and_index
from image_retrieval.model import (
    EmbeddingGemma2,
    Intent,
    SigLIP2,
    create_embedder,
    recipe_fingerprint,
    validate_vectors,
)
from image_retrieval.search import normalize_scores, rank_by_similarity, search

ROOT = Path(__file__).resolve().parents[1]


class FakeEmbedder:
    dimension = 3
    manifest = {"dimension": 3, "recipe": "offline-fake-v1"}

    def __init__(self, fail_at=None, failure=RuntimeError):
        self.batches = []
        self.fail_at = fail_at
        self.failure = failure
        self.closed = False

    @property
    def recipe_id(self):
        return recipe_fingerprint(self.manifest)

    def encode_images(self, images, *, intent="corpus"):
        self.batches.append((len(images), intent))
        if self.fail_at == len(self.batches):
            raise self.failure("injected failure")
        return np.tile(np.array([[1, 0, 0]], dtype=np.float32), (len(images), 1))

    def encode_texts(self, texts, *, intent="query"):
        return self.encode_images(texts, intent=intent)

    def close(self):
        self.closed = True


def image_bytes(color="red", format="PNG", **kwargs):
    buffer = BytesIO()
    Image.new("RGB", (3, 2), color).save(buffer, format=format, **kwargs)
    return buffer.getvalue()


def make_store(tmp_path, embedder=None):
    embedder = embedder or FakeEmbedder()
    return FeatureStore(tmp_path / "cache" / "features.npz", "test", embedder.manifest)


def config_data():
    data = yaml.safe_load((ROOT / "profiles.yaml").read_text())
    # Collection names and machine-local paths are user settings, not test fixtures.
    data["default_corpus"] = "images"
    data["corpora"] = {"images": {"id": "test-images", "root": "photos"}}
    return data


def write_config(tmp_path, data):
    path = tmp_path / "profiles.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_config_repository_relative_and_selection(tmp_path, monkeypatch):
    path = write_config(tmp_path, config_data())
    monkeypatch.chdir(tmp_path.parent)
    settings = load_settings(path, repo_root=tmp_path)
    assert settings.corpora["images"].root == tmp_path / "photos"
    assert settings.cache_root == tmp_path / ".cache/image-retrieval"
    assert settings.select("siglip2")[0].family == "siglip2"
    assert settings.default_profile == "gemma2"
    with pytest.raises(ConfigError):
        settings.select("absent")


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(extra=True),
        lambda d: d.update(schema_version=True),
        lambda d: d["runtime"].update(batch_size=True),
        lambda d: d["runtime"].update(batch_size=0),
        lambda d: d["runtime"].update(batch_size=257),
        lambda d: d["runtime"].update(precision="float16"),
        lambda d: d["runtime"].update(device="cuda"),
        lambda d: d["runtime"].update(offline="false"),
        lambda d: d["runtime"].update(checkpoint_interval=-1),
        lambda d: d["profiles"]["gemma2"].update(revision="main"),
        lambda d: d["profiles"]["gemma2"].update(kwargs={}),
        lambda d: d["profiles"]["gemma2"].update(model="unknown/model"),
        lambda d: d["corpora"]["images"].update(id="../escape"),
        lambda d: d.update(default_corpus="absent"),
    ],
)
def test_invalid_config(tmp_path, change):
    data = config_data()
    change(data)
    with pytest.raises(ConfigError):
        load_settings(write_config(tmp_path, data), repo_root=tmp_path)


def test_yaml_duplicate_unsafe_and_missing_placeholder(tmp_path, monkeypatch):
    path = tmp_path / "profiles.yaml"
    for text in (
        "schema_version: 1\nschema_version: 1",
        "!!python/object:os.system {}",
        "[]",
    ):
        path.write_text(text)
        with pytest.raises(ConfigError):
            load_settings(path, repo_root=tmp_path)
    monkeypatch.delenv("MISSING_IMAGE_TEST", raising=False)
    data = config_data()
    data["corpora"]["images"]["root"] = "${MISSING_IMAGE_TEST}"
    with pytest.raises(ConfigError, match="Missing path placeholder"):
        load_settings(write_config(tmp_path, data), repo_root=tmp_path)


def test_explicit_dotenv_and_environment_precedence(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("LOCAL_TEST_IMAGES=local\n")
    data = config_data()
    data["corpora"]["images"]["root"] = "${LOCAL_TEST_IMAGES}"
    path = write_config(tmp_path, data)
    monkeypatch.delenv("LOCAL_TEST_IMAGES", raising=False)
    assert (
        load_settings(path, repo_root=tmp_path).corpora["images"].root
        == tmp_path / "local"
    )
    assert "LOCAL_TEST_IMAGES" not in __import__("os").environ
    monkeypatch.setenv("LOCAL_TEST_IMAGES", "override")
    assert (
        load_settings(path, repo_root=tmp_path).corpora["images"].root
        == tmp_path / "override"
    )


def test_imports_do_not_load_settings(monkeypatch):
    monkeypatch.delenv("IMAGE_FOLDER", raising=False)
    import image_retrieval.config as config

    monkeypatch.setattr(
        config, "load_settings", lambda *a, **k: pytest.fail("import loaded settings")
    )
    for name in (
        "image_retrieval",
        "model",
        "features",
        "indexing",
        "search",
        "render",
        "notebook",
        "main",
    ):
        importlib.import_module(
            name if name == "image_retrieval" else f"image_retrieval.{name}"
        )


def test_image_hash_rgb_first_frame_orientation_and_detachment(tmp_path):
    captured = image_bytes()
    loaded = load_image(captured)
    assert loaded.content_hash == sha256(captured).hexdigest()
    assert loaded.image.mode == "RGB"
    file = tmp_path / "test.png"
    file.write_bytes(captured)
    assert load_image(file).image.getpixel((0, 0)) == (255, 0, 0)
    exif = Image.Exif()
    exif[274] = 6
    assert load_image(image_bytes(format="JPEG", exif=exif)).image.size == (2, 3)
    stream = BytesIO()
    Image.new("RGB", (2, 2), "red").save(
        stream, "GIF", save_all=True, append_images=[Image.new("RGB", (2, 2), "blue")]
    )
    assert load_image(stream.getvalue()).image.getpixel((0, 0)) == (255, 0, 0)
    assert load_image(image_bytes(format="WEBP")).image.mode == "RGB"


@pytest.mark.parametrize(
    "vectors",
    [
        np.zeros((1, 3), dtype=np.float32),
        np.array([[np.nan, 1, 0]]),
        np.ones(3),
        np.ones((1, 2)),
        np.ones((1, 3), dtype=int),
    ],
)
def test_invalid_vectors(vectors):
    with pytest.raises(ValueError):
        validate_vectors(vectors, 1, 3, normalize=True)


def test_factory_identity_and_empty_batches(monkeypatch):
    import image_retrieval.model as module

    monkeypatch.setattr(module, "get_device", lambda selection: "cpu")
    settings = load_settings(ROOT / "profiles.yaml", repo_root=ROOT)
    for name, expected in (("gemma2", EmbeddingGemma2), ("siglip2", SigLIP2)):
        profile = settings.profiles[name]
        first = create_embedder(profile, Runtime())
        assert isinstance(first, expected)
        assert first.encode_images([]).shape == (0, first.dimension)
        assert first.encode_texts([]).dtype == np.float32
        assert first._model is None
        second = expected(
            profile, Runtime(batch_size=20, offline=True, checkpoint_interval=1), "cpu"
        )
        assert first.recipe_id == second.recipe_id
        assert (
            first.recipe_id
            != expected(replace(profile, revision="a" * 40), Runtime(), "cpu").recipe_id
        )
        assert first.recipe_id != expected(profile, Runtime(), "mps").recipe_id
        assert (
            recipe_fingerprint(dict(reversed(list(first.manifest.items()))))
            == first.recipe_id
        )


def test_gemma_routing_order_and_no_prefix():
    profile = Profile("embeddinggemma2", "google/embeddinggemma-2", "a" * 40, "a" * 40)
    adapter = EmbeddingGemma2(profile, Runtime(batch_size=2), "cpu")
    calls = []

    class Double:
        def encode(self, inputs, **kwargs):
            calls.append((inputs, kwargs))
            result = np.zeros((len(inputs), 768), dtype=np.float32)
            for i in range(len(inputs)):
                result[i, i] = 2
            return result

    adapter._model = Double()
    vectors = adapter.encode_texts(["https://example.com/cat.jpg", "second"])
    assert vectors[0, 0] == vectors[1, 1] == 1
    assert calls[-1][0] == [{"text": "https://example.com/cat.jpg"}, {"text": "second"}]
    assert calls[-1][1]["prompt"] == ""
    images = [Image.new("RGB", (2, 2))]
    adapter.encode_images(images, intent="query")
    assert calls[-1][0] == [{"image": images[0]}]
    assert calls[-1][1]["prompt"] == ""
    with pytest.raises(TypeError):
        adapter.encode_texts([images[0]])
    with pytest.raises(ValueError):
        adapter.encode_images(images, intent=cast(Intent, "other"))


def test_gemma_loader_parameters(monkeypatch):
    import sentence_transformers

    calls = []

    class Double:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

        def float(self):
            return self

        def eval(self):
            return self

        def get_embedding_dimension(self):
            return 768

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", Double)
    profile = Profile("embeddinggemma2", "google/embeddinggemma-2", "a" * 40, "a" * 40)
    adapter = EmbeddingGemma2(profile, Runtime(offline=True), "cpu")
    adapter._load()
    kwargs = calls[0][1]
    assert kwargs["trust_remote_code"] is False
    assert kwargs["local_files_only"] is True
    assert kwargs["config_kwargs"] == {"audio_config": None}
    assert kwargs["revision"] == kwargs["processor_kwargs"]["revision"] == "a" * 40
    import torch

    assert kwargs["model_kwargs"]["torch_dtype"] == torch.float32


def test_siglip_modeloutput_processor_and_bounded_batches():
    import torch

    profile = Profile(
        "siglip2", "google/siglip2-so400m-patch14-384", "a" * 40, "b" * 40
    )
    adapter = SigLIP2(profile, Runtime(batch_size=2), "cpu")
    calls = []

    class Inputs(dict):
        def to(self, device):
            return self

    def processor(**kwargs):
        calls.append(kwargs)
        return Inputs(n=len(kwargs.get("text", kwargs.get("images"))))

    class Model:
        def get_text_features(self, n, return_dict):
            assert return_dict
            return SimpleNamespace(pooler_output=torch.ones((n, 1152)))

        get_image_features = get_text_features

    adapter._model, adapter._processor = Model(), processor
    vectors = adapter.encode_texts(["first", "second", "third"])
    assert vectors.shape == (3, 1152) and vectors.dtype == np.float32
    assert [len(call["text"]) for call in calls] == [2, 1]
    assert calls[0]["padding"] == "max_length" and calls[0]["max_length"] == 64
    adapter.encode_images([Image.new("RGB", (2, 2))], intent="query")
    assert "text" not in calls[-1] and "images" in calls[-1]


def test_store_roundtrip_empty_duplicate_paths_and_identity(tmp_path):
    store = make_store(tmp_path)
    store.save()
    loaded = FeatureStore.load(store.path, "test", store.manifest)
    assert loaded.values.shape == (0, 3) and loaded.complete is False
    content_hash = "a" * 64
    store.add(content_hash, np.array([1, 0, 0], dtype=np.float32))
    store.membership = {"a.png": content_hash, "b.png": content_hash}
    store.complete = True
    store.save()
    loaded = FeatureStore.load(store.path, "test", store.manifest)
    assert loaded.keys == ["a.png", "b.png"]
    assert loaded.values.shape == (2, 3) and len(loaded.vectors) == 1
    assert loaded.complete
    with np.load(store.path, allow_pickle=False) as archive:
        assert all(archive[name].dtype.kind != "O" for name in archive.files)
    with pytest.raises(StoreError, match="recipe mismatch"):
        FeatureStore.load(store.path, "test", {"dimension": 3, "recipe": "different"})
    with pytest.raises(StoreError):
        FeatureStore.load(store.path, "wrong", store.manifest)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda a: a.update(vectors=np.ones((2, 3), dtype=np.float32)),
        lambda a: a.update(vectors=np.array([[np.nan, 0, 0]], dtype=np.float32)),
        lambda a: a.update(vectors=np.array([[1, 0, 0]], dtype=np.float64)),
        lambda a: a.update(paths=np.array(["../escape.png"])),
        lambda a: a.update(path_hashes=np.array(["b" * 64])),
        lambda a: a.update(hashes=np.array(["bad"])),
        lambda a: a.update(
            paths=np.array(["a.png", "a.png"]),
            path_hashes=np.array(["a" * 64, "a" * 64]),
        ),
        lambda a: a.update(paths=np.array(["a.png"], dtype=object)),
    ],
)
def test_store_malformed_arrays(tmp_path, mutation):
    store = make_store(tmp_path)
    store.add("a" * 64, np.array([1, 0, 0], dtype=np.float32))
    store.membership["a.png"] = "a" * 64
    store.save()
    with np.load(store.path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    mutation(arrays)
    np.savez(store.path, **arrays)
    with pytest.raises(StoreError):
        FeatureStore.load(store.path, "test", store.manifest)


def test_legacy_preserved_missing_and_atomic_failure(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    with pytest.raises(FileNotFoundError):
        FeatureStore.load(store.path, "test", store.manifest)
    assert not FeatureStore.load(
        store.path, "test", store.manifest, missing_ok=True
    ).keys
    store.save()
    committed = store.path.read_bytes()
    import image_retrieval.features as module

    monkeypatch.setattr(
        module.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("injected"))
    )
    with pytest.raises(OSError):
        store.save()
    assert store.path.read_bytes() == committed
    assert list(store.path.parent.iterdir()) == [store.path]
    np.savez(store.path, np.array({"legacy": np.ones(3)}, dtype=object))
    original = store.path.read_bytes()
    with pytest.raises(StoreError, match="legacy"):
        FeatureStore.load(store.path, "test", store.manifest)
    assert store.path.read_bytes() == original


def test_single_writer_lock(tmp_path):
    path = tmp_path / "features.npz"
    with writer_lock(path):
        with pytest.raises(StoreError, match="Another writer"):
            with writer_lock(path):
                pass
    with writer_lock(path):
        pass


def test_reconcile_new_noop_changed_renamed_deleted_duplicate_and_webp(tmp_path):
    root = tmp_path / "photos"
    root.mkdir()
    (root / "a.png").write_bytes(image_bytes())
    (root / "copy.PNG").write_bytes(image_bytes())
    (root / "b.webp").write_bytes(image_bytes("blue", format="WEBP"))
    embedder, store = FakeEmbedder(), make_store(tmp_path)
    report = encode_and_index(embedder, root, store, batch_size=2)
    assert report.generated == 2 and len(store.keys) == 3
    assert max(size for size, _ in embedder.batches) <= 2
    embedder.batches.clear()
    assert encode_and_index(embedder, root, store).generated == 0
    assert not embedder.batches
    (root / "a.png").rename(root / "renamed.png")
    (root / "copy.PNG").unlink()
    assert encode_and_index(embedder, root, store).generated == 0
    assert store.keys == ["b.webp", "renamed.png"]
    (root / "renamed.png").write_bytes(image_bytes("green"))
    assert encode_and_index(embedder, root, store).generated == 1
    (root / "renamed.png").write_bytes(b"broken")
    report = encode_and_index(embedder, root, store)
    assert report.skipped == ["renamed.png"] and store.keys == ["b.webp"]


def test_progress_tracks_handled_paths_after_validated_batches(tmp_path):
    root = tmp_path / "photos"
    root.mkdir()
    for name, color in (("a.png", "red"), ("b.png", "red"), ("c.png", "blue")):
        (root / name).write_bytes(image_bytes(color))
    (root / "d.png").write_bytes(b"broken")
    store = make_store(tmp_path)
    events = []

    def progress(completed, total, report):
        events.append(
            (completed, total, report.generated, report.reused, len(report.skipped))
        )

    encode_and_index(
        FakeEmbedder(), root, store, batch_size=2, report_progress=progress
    )
    assert events[0] == (0, 4, 0, 0, 0)
    assert events[-1] == (4, 4, 2, 1, 1)
    # Pending duplicates only advance once the shared embedding is validated.
    assert not any(0 < event[0] < 3 for event in events)
    events.clear()
    encode_and_index(FakeEmbedder(), root, store, report_progress=progress)
    assert events[-1] == (4, 4, 0, 3, 1)


def test_checkpoints_count_successful_unique_vectors(tmp_path, monkeypatch):
    root = tmp_path / "photos"
    root.mkdir()
    for i in range(5):
        (root / f"{i}.png").write_bytes(image_bytes((i, 0, 0)))
    (root / "broken.png").write_bytes(b"broken")
    store = make_store(tmp_path)
    saved = []
    original = store.save

    def save():
        saved.append((len(store.vectors), store.complete))
        original()

    monkeypatch.setattr(store, "save", save)
    encode_and_index(FakeEmbedder(), root, store, batch_size=3, checkpoint_interval=2)
    assert saved == [(2, False), (4, False), (5, True)]


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_failure_and_interruption_preserve_resume_and_old_membership(tmp_path, failure):
    root = tmp_path / "photos"
    root.mkdir()
    store = make_store(tmp_path)
    store.add("a" * 64, np.array([1, 0, 0], dtype=np.float32))
    store.membership["deleted.png"] = "a" * 64
    for i in range(3):
        (root / f"{i}.png").write_bytes(image_bytes((i, 0, 0)))
    embedder = FakeEmbedder(fail_at=2, failure=failure)
    with pytest.raises(failure):
        encode_and_index(embedder, root, store)
    loaded = FeatureStore.load(store.path, "test", store.manifest)
    assert (
        not loaded.complete and "deleted.png" in loaded.keys and "0.png" in loaded.keys
    )
    report = encode_and_index(FakeEmbedder(), root, loaded)
    assert report.generated == 2 and loaded.complete
    assert loaded.keys == ["0.png", "1.png", "2.png"]


def test_invalid_inference_batch_never_published(tmp_path, monkeypatch):
    root = tmp_path / "photos"
    root.mkdir()
    (root / "a.png").write_bytes(image_bytes())
    store = make_store(tmp_path)
    embedder = FakeEmbedder()
    monkeypatch.setattr(
        embedder, "encode_images", lambda *a, **k: np.zeros((1, 3), dtype=np.float32)
    )
    with pytest.raises(ValueError):
        encode_and_index(embedder, root, store)
    assert not store.keys and not store.vectors and not store.complete
    assert not FeatureStore.load(store.path, "test", store.manifest).complete


def test_scores_and_checked_search(tmp_path):
    assert normalize_scores(np.array([])).size == 0
    assert normalize_scores(np.array([3])).tolist() == [0]
    assert normalize_scores(np.array([3, 3])).tolist() == [0, 0]
    assert normalize_scores(np.array([1, 2, 3])).tolist() == [-1, 0, 1]
    matrix = np.eye(3, dtype=np.float32)
    assert rank_by_similarity(matrix, matrix[0], 0)[0].size == 0
    assert rank_by_similarity(matrix, matrix[0], 99)[0].tolist() == [0, 1, 2]
    assert rank_by_similarity(np.empty((0, 3)), matrix[0])[0].size == 0
    with pytest.raises(ValueError):
        rank_by_similarity(matrix, matrix[0], -1)
    store = make_store(tmp_path)
    embedder = FakeEmbedder()
    assert search(store, embedder, "query")[0].size == 0
    embedder.manifest = {"dimension": 3, "recipe": "incompatible"}
    with pytest.raises(StoreError):
        search(store, embedder, "query")


def test_cli_selectors_statuses_and_errors(tmp_path, monkeypatch, capsys):
    import image_retrieval.main as cli

    data = config_data()
    root = tmp_path / "photos"
    root.mkdir()
    (root / "a.png").write_bytes(image_bytes())
    settings = load_settings(write_config(tmp_path, data), repo_root=tmp_path)
    monkeypatch.setattr(cli, "load_settings", lambda path: settings)
    monkeypatch.setattr(cli, "create_embedder", lambda *a: FakeEmbedder())
    assert cli.run(["--profile", "siglip2", "--corpus", "images"]) == 0
    _output = capsys.readouterr()
    assert "Indexing images" in _output.err
    assert "100%" in _output.err
    assert "encoded=1" in _output.err
    (root / "bad.png").write_bytes(b"invalid")
    assert cli.run([]) == 2
    assert "Skipped bad.png" in capsys.readouterr().err
    assert cli.run(["--profile", "missing"]) == 1
    monkeypatch.setattr(
        cli,
        "create_embedder",
        lambda *a: FakeEmbedder(fail_at=1, failure=KeyboardInterrupt),
    )
    (root / "a.png").write_bytes(image_bytes("blue"))
    assert cli.run([]) == 130
    with pytest.raises(SystemExit) as exc:
        cli.run(["--rebuild"])
    assert exc.value.code == 2


def test_notebook_lifecycle_read_only_and_missing_original(tmp_path, monkeypatch):
    import image_retrieval.notebook as module
    from image_retrieval.render import render_result_card

    settings = load_settings(write_config(tmp_path, config_data()), repo_root=tmp_path)
    created = []

    def factory(*args):
        if created:
            assert created[-1].closed
        created.append(FakeEmbedder())
        return created[-1]

    monkeypatch.setattr(module, "create_embedder", factory)
    session = module.NotebookSession(settings)
    session.select("gemma2", "images")
    session.select("siglip2", "images")
    assert created[0].closed and session.store is None
    with pytest.raises(FileNotFoundError):
        session.reload()
    assert "--profile siglip2" in session.indexing_guidance()
    assert not settings.cache_root.exists()
    assert "Original unavailable" in render_result_card("missing.png", root=tmp_path)
    assert "&lt;" in render_result_card("<unsafe>.png", root=tmp_path)
