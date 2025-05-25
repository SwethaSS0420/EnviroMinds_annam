"""
Team Name: EnviroMinds
Team Members: Sanjana Sudarsan, Swetha Sriram, Lohithaa K M
Leaderboard Rank: 29

This file handles post-processing for the soil classification task.
It includes:
- Model ensembling
- Threshold selection using F1 score
- Saving predictions to submission.csv
"""

# Here you add all the post-processing related details for the task completed from Kaggle.

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

def ensemble_scores(X, iso_models, ocsvm_models, scalers):
    iso_scores = np.mean([m.decision_function(X) for m in iso_models], axis=0)
    ocsvm_scores = np.mean([m.decision_function(s.transform(X)) for m, s in zip(ocsvm_models, scalers)], axis=0)
    return (iso_scores + ocsvm_scores) / 2

def tune_threshold(y_true, scores, num_steps=100):
    best_f1 = 0
    best_thresh = None
    for t in np.linspace(min(scores), max(scores), num_steps):
        preds = (scores >= t).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_thresh, best_f1

def save_submission(image_ids, predictions, filename="submission.csv"):
    submission = pd.DataFrame({
        "image_id": image_ids,
        "soil_type": predictions
    })
    submission.to_csv(filename, index=False)
    print(f"Saved to {filename}")

# Optional wrapper for CLI or modular use
def postprocessing(X_val, y_val, X_test, test_image_ids, iso_models, ocsvm_models, scalers):
    print("Running postprocessing...")

    # Combine scores
    val_scores = ensemble_scores(X_val, iso_models, ocsvm_models, scalers)

    # Threshold tuning
    best_thresh, best_f1 = tune_threshold(y_val, val_scores)
    print(f"Best F1: {best_f1:.4f} at threshold {best_thresh:.4f}")

    # Final prediction on test set
    test_scores = ensemble_scores(X_test, iso_models, ocsvm_models, scalers)
    test_preds = (test_scores >= best_thresh).astype(int)

    # Save submission
    save_submission(test_image_ids, test_preds)
    return test_preds, best_thresh, best_f1