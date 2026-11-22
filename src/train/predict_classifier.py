from __future__ import annotations

import argparse
import json
from pathlib import Path

import timm
import torch
from PIL import Image
from torchvision import transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict empty/exist/sleeping for one image."
    )
    parser.add_argument("image_path", type=Path)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/sleep_classifier_ensemble.pt"),
    )
    return parser.parse_args()


def _build_model(model_name: str, num_classes: int, device: torch.device):
    return (
        timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        .to(device)
        .eval()
    )


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    labels = checkpoint["labels"]
    image_size = checkpoint["image_size"]

    # An ensemble bundle holds several state_dicts under "members"; a plain
    # checkpoint has a single "state_dict". Both are softmax-averaged uniformly
    # (a single model is just the 1-member case).
    if checkpoint.get("ensemble"):
        state_dicts = checkpoint["members"]
    else:
        state_dicts = [checkpoint["state_dict"]]

    model = _build_model(checkpoint["model_name"], len(labels), device)

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    image = Image.open(args.image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        prob_sum = None
        for state_dict in state_dicts:
            model.load_state_dict(state_dict)
            member_probs = torch.softmax(model(tensor), dim=1)
            prob_sum = member_probs if prob_sum is None else prob_sum + member_probs
        probs = (prob_sum / len(state_dicts))[0].cpu().tolist()

    payload = {
        "image_path": str(args.image_path),
        "predicted_label": labels[int(torch.tensor(probs).argmax().item())],
        "probabilities": {label: float(prob) for label, prob in zip(labels, probs)},
        "ensemble_members": len(state_dicts),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
