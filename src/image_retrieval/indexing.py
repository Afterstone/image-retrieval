from pathlib import Path

import tqdm
from PIL import Image

from .features import FeatureStore
from .search import encode_image

IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]


def discover_images(
    folder: Path,
    extensions: list[str] = IMAGE_EXTENSIONS,
    known_paths: set[str] | None = None,
) -> list[str]:
    image_files: list[str] = []
    for ext in extensions:
        image_files.extend(str(f) for f in folder.glob(f"**/{ext}") if f.is_file())

    if known_paths:
        image_files = [f for f in image_files if f not in known_paths]

    return image_files


def encode_and_index(
    model,
    preprocess,
    device,
    image_files: list[str],
    store: FeatureStore,
    checkpoint_every: int = 500,
) -> None:
    for i, image_file in tqdm.tqdm(enumerate(image_files), total=len(image_files)):
        try:
            image = Image.open(image_file).convert("RGB")
        except Exception:
            try:
                image = Image.open(image_file).convert("RGBA")
            except Exception as e:
                print(f"Error loading image {image_file}: {e}")
                continue

        vector = encode_image(model, preprocess, device, image)
        store.update(image_file, vector)

        if i % checkpoint_every == 0:
            store.save()

    store.save()
