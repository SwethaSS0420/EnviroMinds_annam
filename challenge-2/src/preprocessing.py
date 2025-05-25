"""
Team Name: EnviroMinds
Team Members: Sanjana Sudarsan, Swetha Sriram, Lohithaa K M
Leaderboard Rank: 30

This file handles all preprocessing tasks including:
- Reading dataset paths and labels
- Defining image transformations
- Extracting features using EfficientNet-B3
"""
# Here you add all the post-processing related details for the task completed from Kaggle.

import os
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torchvision import transforms, models

# Define transformations
normal_tf = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

anomaly_tf = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.RandomAffine(45, scale=(0.4, 1.5), shear=30),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.9, contrast=0.9, saturation=0.9, hue=0.2),
    transforms.GaussianBlur(5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load EfficientNet-B3 feature extractor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT).to(device)
feature_extractor = torch.nn.Sequential(*list(base_model.children())[:-1]).eval()

def get_image_paths(image_ids, dir_path):
    return [os.path.join(dir_path, img_id) for img_id in image_ids]

def extract_features(image_paths, transform):
    features = []
    for path in tqdm(image_paths, desc="Extracting features"):
        img = Image.open(path).convert("RGB")
        img_t = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = feature_extractor(img_t).view(1, -1).cpu().numpy()
        features.append(feat[0])
    return np.array(features)

def load_data(train_csv, test_csv, train_dir, test_dir):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    train_ids = train_df["image_id"].tolist()
    test_ids = test_df["image_id"].tolist()
    train_paths = get_image_paths(train_ids, train_dir)
    test_paths = get_image_paths(test_ids, test_dir)
    return train_df, test_df, train_paths, test_paths