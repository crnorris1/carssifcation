import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
import time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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
train_loader = DataLoader(train_dataset, batch_size=16, num_workers=8, pin_memory=True, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=16, num_workers=8, pin_memory=True)

# set device (cpu or gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create 3d pca visualization
def pca_3d_visualization(model, data_loader, num_samples=500):
    model.eval()
    all_imgs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            if len(all_imgs) >= num_samples:
                break

            batch_size = images.size(0)
            take = min(num_samples - len(all_imgs), batch_size)

            imgs_batch = images[:take].to(device)
            labels_batch = labels[:take]

            outputs = model(imgs_batch)
            _, preds_batch = torch.max(outputs, 1)

            all_imgs.append(imgs_batch.cpu())
            all_preds.append(preds_batch.cpu())
            all_labels.append(labels_batch.cpu())

    if not all_imgs:
        print("No images collected for PCA.")
        return

    imgs = torch.cat(all_imgs, dim=0)
    preds = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # Flatten images
    imgs_flat = imgs.view(imgs.size(0), -1).numpy()

    # 2 components PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(imgs_flat)

    # 3D scatter plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        X_pca[:, 0],  # PCA component 1 -> x-axis (input variation)
        X_pca[:, 1],  # PCA component 2 -> y-axis
        labels.numpy(),
        c=preds.numpy(),
        cmap="tab10",
        s=10,
        alpha=0.7
    )

    ax.set_xlabel("PCA 1 (input variation)")
    ax.set_xlabel("PCA 2 (input variation)")
    ax.set_zlabel("Predicted class")
    ax.set_title("3D PCA Visualization of Model Predictions")

    plt.tight_layout()
    plt.show()

# calculate the accuracy given a data set
def calc_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total if total > 0 else 0
    return acc

def train(num_epochs=10, weights_path="car_classifier_weights.pth"):
    print(f"Using device: {device}")
    num_classes = len(train_dataset.classes)
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    class_counts = torch.tensor([44, 82, 22], dtype=torch.float, device=device)
    class_weights = 1.0 / class_counts
    criterion = nn.CrossEntropyLoss(weight=class_weights)
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
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Accuracy: {calc_accuracy(model, train_loader)*100:.2f}%")

    print(f"Training Accuracy after training: {calc_accuracy(model, train_loader)*100:.2f}%")
    print(f"Validation Accuracy after training: {calc_accuracy(model, val_loader)*100:.2f}%")

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

if __name__ == "__main__":
    train()
    model = load_model_for_inference("car_classifier_weights.pth")
    pca_3d_visualization(model, val_loader, num_samples=500)


