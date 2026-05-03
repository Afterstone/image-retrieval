from .config import FEATURE_FILE, IMAGE_FOLDER
from .device import get_device
from .features import FeatureStore
from .indexing import discover_images, encode_and_index
from .model import load_siglip2


def main() -> None:
    device = get_device()
    print(f"Device: {device}")

    print("Loading model...")
    model, preprocess, _ = load_siglip2(device=device)

    store = FeatureStore.load(FEATURE_FILE)

    image_files = discover_images(IMAGE_FOLDER, known_paths=set(store.keys))
    print(f"Found {len(image_files)} new images to index.")

    encode_and_index(model, preprocess, device, image_files, store)


if __name__ == "__main__":
    main()
