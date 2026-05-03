import torch
from open_clip import create_model_from_pretrained, get_tokenizer

MODEL_ID = "hf-hub:timm/ViT-SO400M-14-SigLIP2"


def load_siglip2(
    model_id: str = MODEL_ID,
    device: torch.device | None = None,
) -> tuple:
    from .device import get_device

    if device is None:
        device = get_device()

    model, preprocess = create_model_from_pretrained(model_id)
    model = model.to(device)
    tokenizer = get_tokenizer(model_id)
    return model, preprocess, tokenizer
