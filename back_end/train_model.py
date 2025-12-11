import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights

# Resize images
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load data
train_dataset = datasets.ImageFolder("dataset/train", transform=train_transforms)
val_dataset   = datasets.ImageFolder("dataset/val", transform=val_transforms)

# training and validation dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(num_epochs=10, weights_path="car_classifier_weights.pth"):
    print(f"Using device: {device}")
    num_classes = len(train_dataset.classes)
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # optional: validation loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total if total > 0 else 0
        print(f"Epoch {epoch+1}/{num_epochs}, val acc: {acc:.4f}")

    torch.save(model.state_dict(), weights_path)
    print(f"Model saved to {weights_path}")
    return model

def load_model_for_inference(weights_path):
    num_classes = len(train_dataset.classes)
    weights = ResNet18_Weights.DEFAULT
    mdl = models.resnet18(weights=weights)
    mdl.fc = nn.Linear(mdl.fc.in_features, num_classes)
    mdl.load_state_dict(torch.load(weights_path, map_location=device))
    mdl.to(device)
    mdl.eval()
    return mdl

def preprocess_image(image_path):
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    tensor = val_transforms(img).unsqueeze(0)
    return tensor.to(device)

def predict_image(image_path, model, class_names):
    tensor = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(tensor)
        _, pred_idx = torch.max(outputs, 1)
        pred_idx = pred_idx.item()
    pred_class = class_names[pred_idx]
    return pred_class, pred_idx

if __name__ == "__main__":
    # training happens only when running train_model.py directly
    train()


