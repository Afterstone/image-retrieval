"""Content reconciliation with bounded inference and coherent checkpoints."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from PIL import Image

from .features import FeatureStore
from .images import IMAGE_EXTENSIONS, load_image
from .model import Embedder, validate_vectors


def discover_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise ValueError(f"Corpus root is not a directory: {folder}")

    def fail(error):
        raise error

    files = []
    # Unlike rglob, walk with onerror never silently reconciles an unreadable subtree.
    for directory, dirs, names in os.walk(folder, onerror=fail, followlinks=False):
        dirs.sort()
        for name in sorted(names):
            path = Path(directory) / name
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                files.append(path)
    return sorted(files, key=lambda path: path.relative_to(folder).as_posix())


@dataclass
class IndexReport:
    generated: int = 0
    reused: int = 0
    skipped: list[str] = field(default_factory=list)


def encode_and_index(
    embedder: Embedder,
    root: Path,
    store: FeatureStore,
    *,
    batch_size: int = 1,
    checkpoint_interval: int = 500,
    report_skip: Callable[[str], None] | None = None,
    report_progress: Callable[[int, int, IndexReport], None] | None = None,
) -> IndexReport:
    """Hold the store writer lock across loading and this operation.

    Until completion old membership is retained alongside verified new paths.
    Failed/unreadable current paths are removed only on completed traversal.
    Unique vectors survive interruption even when their paths are not committed.
    """
    if (
        type(batch_size) is not int
        or not 1 <= batch_size <= 256
        or type(checkpoint_interval) is not int
        or checkpoint_interval < 1
    ):
        raise ValueError("Invalid batch/checkpoint size")
    store.assert_compatible(embedder.manifest)
    report = IndexReport()
    verified: dict[str, str] = {}
    pending: dict[str, tuple[Image.Image, list[str]]] = {}
    since_checkpoint = 0
    store.complete = False
    total = 0

    def progress():
        if report_progress:
            report_progress(len(verified) + len(report.skipped), total, report)

    def flush():
        nonlocal since_checkpoint
        if not pending:
            return
        hashes = list(pending)
        images = [pending[h][0] for h in hashes]
        try:
            vectors = validate_vectors(
                embedder.encode_images(images, intent="corpus"),
                len(hashes),
                store.dimension,
            )
            # Entire batch validated before any vector is published.
            for content_hash, vector in zip(hashes, vectors, strict=True):
                store.add(content_hash, vector)
                for path in pending[content_hash][1]:
                    verified[path] = content_hash
                    store.membership[path] = content_hash
            report.generated += len(hashes)
            since_checkpoint += len(hashes)
            pending.clear()
            progress()
            if since_checkpoint == checkpoint_interval:
                store.save()
                since_checkpoint = 0
        finally:
            for image in images:
                image.close()

    try:
        candidates = discover_images(root)
        total = len(candidates)
        progress()
        for path in candidates:
            relative = path.relative_to(root).as_posix()
            try:
                loaded = load_image(path)
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                report.skipped.append(relative)
                if report_skip:
                    report_skip(f"Skipped {relative}: {exc}")
                progress()
                continue
            content_hash, image = loaded.content_hash, loaded.image
            if content_hash in store.vectors:
                verified[relative] = content_hash
                store.membership[relative] = content_hash
                report.reused += 1
                image.close()
                progress()
            elif content_hash in pending:
                pending[content_hash][1].append(relative)
                report.reused += 1
                image.close()
            else:
                pending[content_hash] = (image, [relative])
                if len(pending) >= min(
                    batch_size, checkpoint_interval - since_checkpoint
                ):
                    flush()
        flush()
    except BaseException:
        # On inference failure or Ctrl-C publish only validated work; never reconcile deletions.
        for image, _ in pending.values():
            image.close()
        store.save()
        raise
    store.membership = verified
    store.complete = True
    store.save()
    return report
