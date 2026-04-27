import torch
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer
from pathlib import Path
import time
import tqdm
import numpy as np

# 1. Device selection
print("Setting device...")
device = (
    torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
)

# 2. Move model to device
print("Loading model...")
model, preprocess = create_model_from_pretrained("hf-hub:timm/ViT-SO400M-14-SigLIP2")  # type: ignore
model = model.to(device)

print("Loading tokenizer...")
tokenizer = get_tokenizer("hf-hub:timm/ViT-SO400M-14-SigLIP2")

base_folder = Path("/Users/rikard/Google Drive/Memes")
image_feature_file = base_folder / "__siglip2_image_features.npz"

if image_feature_file.exists():
    all_image_features = np.load(image_feature_file, allow_pickle=True)["arr_0"].item()
else:
    all_image_features = {}

# Find all image files in the base folder
image_file_types = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]
image_files = []
for file_type in image_file_types:
    image_files.extend(base_folder.glob(f"**/{file_type}"))

image_files = [
    str(image_file)
    for image_file in image_files
    if str(image_file) not in all_image_features and image_file.is_file()
]

for i, image_file in tqdm.tqdm(enumerate(image_files), total=len(image_files)):
    try:
        image = Image.open(image_file).convert("RGB")
    except Exception as e:
        try:
            image = Image.open(image_file).convert("RGBA")
        except Exception as e:
            print(f"Error loading image {image_file}: {e}")
            continue

    image = preprocess(image).unsqueeze(0).to(device)  # type: ignore
    with torch.no_grad(), torch.autocast(device.type):
        image_features = model.encode_image(image, normalize=True)  # type: ignore

        all_image_features[image_file] = image_features.detach().squeeze().cpu().numpy()  # type: ignore

    if i % 500 == 0:
        # Save the image features to a file
        np.savez(image_feature_file, all_image_features)  # type: ignore

np.savez(image_feature_file, all_image_features)  # type: ignore
