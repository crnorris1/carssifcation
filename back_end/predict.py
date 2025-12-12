import io
import torch
from PIL import Image

from train_model import load_model_for_inference, train_dataset, val_transforms


def predict(raw_img_bytes: bytes) -> str:
    # 1. Load image from raw bytes
    img = Image.open(io.BytesIO(raw_img_bytes)).convert("RGB")

    # 2. Preprocess using the same validation transforms
    tensor = val_transforms(img).unsqueeze(0)

    # 3. Load model for inference
    inference_model = load_model_for_inference("car_classifier_weights.pth")
    inference_model.eval()

    # 4. Get device of model parameters (cpu or cuda) and move tensor to it
    device = next(inference_model.parameters()).device
    tensor = tensor.to(device)

    class_names = train_dataset.classes

    # 5. Run inference
    with torch.no_grad():
        outputs = inference_model(tensor)
        _, pred_idx = torch.max(outputs, 1)
        pred_idx = pred_idx.item()

    pred_class = class_names[pred_idx]
    return pred_class
