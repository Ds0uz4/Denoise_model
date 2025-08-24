import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Define the global constants
CLASS_NAMES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
CLASSES = {name.lower(): idx for idx, name in enumerate(CLASS_NAMES)}
IMAGE_SIZE = (128, 128)

def load_data(directory, is_labeled=False, is_normalize=False):
    """
    Loads and preprocesses image data from a specified directory.
    
    Args:
        directory (str): The path to the dataset directory.
        is_labeled (bool): True if loading labeled data (e.g., training data with subfolders).
        is_normalize (bool): True to normalize images using a standard mean and std.

    Returns:
        tuple: A tuple containing image data (as a NumPy array) and labels (if is_labeled is True).
               Also returns a list of filenames for test data.
    """
    images = []
    labels = []
    filenames = []

    for root, dirs, files in os.walk(directory):
        files = sorted(files)
        for file in files:
            if not file.lower().endswith('.png'):
                continue

            filepath = os.path.join(root, file)
            img = cv2.imread(filepath, cv2.IMREAD_COLOR)

            if img is None:
                print(f"Warning: failed to read {filepath}")
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMAGE_SIZE)

            img = img.astype(np.float32) / 255.0

            if is_normalize:
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
                img = (img - mean) / std
            
            img = np.transpose(img, (2, 0, 1))

            images.append(img)
            filenames.append(file)

            if is_labeled:
                label_name = os.path.basename(root).lower().strip()
                if label_name in CLASSES:
                    labels.append(CLASSES[label_name])
                else:
                    print(f"Warning: Directory name '{label_name}' not in CLASSES; skipping label.")

    images = np.asarray(images, dtype=np.float32)
    if is_labeled:
        labels = np.asarray(labels, dtype=np.int64)
    else:
        labels = None

    print(f"Loaded {len(images)} images from {directory}.")
    return images, labels, filenames