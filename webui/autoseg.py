"""Automatic foreground segmentation for uploaded inputs.

Reuses briaai/RMBG-1.4 (a background-removal model) through the transformers
dependency that is already installed and whose weights the ActionMesh download
fetches, so no extra packages or models are needed.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        import torch
        from transformers import AutoModelForImageSegmentation

        model = AutoModelForImageSegmentation.from_pretrained(
            "briaai/RMBG-1.4", trust_remote_code=True, local_files_only=True
        )
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        _MODEL = model
    return _MODEL


def warmup() -> None:
    """Load the model now so the first run never fetches or initialises it."""
    _get_model()


def predict_mask(image: Image.Image) -> Image.Image:
    """Return an L-mode mask (0/255) flagged on the largest foreground region."""
    import torch
    from torchvision import transforms

    model = _get_model()
    device = next(model.parameters()).device
    tf = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]),
    ])
    inp = tf(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        # BriaRMBG.forward returns (sigmoid_maps, hidden_states); the maps are
        # already sigmoid-activated, so take the finest side output directly.
        pred = model(inp)[0][0].cpu()[0].squeeze()
    small = pred.unsqueeze(0).unsqueeze(0)
    resized = transforms.functional.resize(small, (image.height, image.width))[0, 0]
    arr = (resized.numpy() > 0.5).astype(np.uint8)
    return Image.fromarray(arr * 255, mode="L")
