"""
Team Name: EnviroMinds
Team Members: Sanjana Sudarsan, Swetha Sriram, Lohithaa K M
Leaderboard Rank: 53

"""

# Here you add all the post-processing related details for the task completed from Kaggle.

import os
import torch
import numpy as np
import pandas as pd
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader
from preprocessing import TestDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_CLASSES = 4
OUTPUT_DIR = "/kaggle/working"

idx2label = {0: 'Alluvial soil', 1: 'Black Soil', 2: 'Clay soil', 3: 'Red soil'}

def inference(test_ids_csv, test_img_dir, n_folds=5):
    test_ids = pd.read_csv(test_ids_csv)
    test_list = test_ids['image_id'].tolist()

    transform_val = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_loader = DataLoader(TestDataset(test_list, test_img_dir, transform_val),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    ensemble_probs = {img_id: [] for img_id in test_list}

    for fold in range(n_folds):
        model = EfficientNet.from_name('efficientnet-b3')
        model._fc = torch.nn.Linear(model._fc.in_features, NUM_CLASSES)
        model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, f"model_fold{fold}.pth")))
        model = model.to(DEVICE).eval()

        with torch.no_grad():
            for imgs, img_ids in test_loader:
                imgs = imgs.to(DEVICE)
                probs = model(imgs).cpu().softmax(1).numpy()
                for i in range(len(img_ids)):
                    ensemble_probs[img_ids[i]].append(probs[i])

    out = []
    for img_id in test_list:
        avg = np.mean(ensemble_probs[img_id], axis=0)
        out.append({
            'image_id': img_id,
            'soil_type': idx2label[int(avg.argmax())]
        })

    pd.DataFrame(out).to_csv(os.path.join(OUTPUT_DIR, "submission.csv"), index=False)
    print("submission.csv written")
