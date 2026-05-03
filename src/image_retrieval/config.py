import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

IMAGE_FOLDER = Path(os.environ["IMAGE_FOLDER"])
FEATURE_FILE_NAME = os.getenv("FEATURE_FILE_NAME", "__siglip2_image_features.npz")
FEATURE_FILE = IMAGE_FOLDER / FEATURE_FILE_NAME
MODEL_ID = os.getenv("MODEL_ID", "hf-hub:timm/ViT-SO400M-14-SigLIP2")
