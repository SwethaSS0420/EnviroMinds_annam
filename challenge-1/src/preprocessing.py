"""
Team Name: EnviroMinds
Team Members: Sanjana Sudarsan, Swetha Sriram, Lohithaa K M
Leaderboard Rank: 53

"""

# Here you add all the preprocessing related details for the task completed from Kaggle.

import os
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
# CLAHE 
def load_clahe_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (300, 300))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    rgb = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb)

# Dataset
class SoilDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = load_clahe_image(os.path.join(self.img_dir, row['image_id']))
        x = self.transform(img)
        y = row['soil_type_idx']
        return x, y
      
class TestDataset(Dataset):
    def __init__(self, image_ids, img_dir, transform):
        self.image_ids = image_ids
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.image_ids)
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img = load_clahe_image(os.path.join(self.img_dir, img_id))
        return self.transform(img), img_id
