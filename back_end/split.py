import os
import shutil
import random

# Paths
source_dir = "dataset/all_categories"
train_dir = "dataset/train"
val_dir   = "dataset/val"

# Create train/val directories
for split_dir in [train_dir, val_dir]:
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)

# Remove existing train/val content (if any) and recreate empty directories
for split_dir in [train_dir, val_dir]:
    if os.path.exists(split_dir):
        shutil.rmtree(split_dir)
    os.makedirs(split_dir, exist_ok=True)

# Set random seed for reproducibility
random.seed(42)

# Percentage for training
train_ratio = 0.8

# Loop through each class
for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    # List all images in class
    images = os.listdir(class_path)
    random.shuffle(images)

    # Split
    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    val_images   = images[split_idx:]

    # Create class subfolders in train/val
    for split, img_list in zip([train_dir, val_dir], [train_images, val_images]):
        class_split_dir = os.path.join(split, class_name)
        os.makedirs(class_split_dir, exist_ok=True)
        for img_name in img_list:
            src_path = os.path.join(class_path, img_name)
            dst_path = os.path.join(class_split_dir, img_name)
            shutil.copy2(src_path, dst_path)

print("Dataset successfully split into train and val!")
