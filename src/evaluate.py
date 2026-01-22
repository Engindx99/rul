import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn.metrics import r2_score

from src.data.loader import load_data
from src.data.preprocess import process_data
from src.data.dataset import CMAPSSDataset
from src.models.lstm import RULPredictor
from src.train import load_config
from src.utils.logger import get_logger

logger = get_logger("Eval")

def compute_metrics(true_rul, pred_rul):
    # RMSE
    mse = np.mean((true_rul - pred_rul) ** 2)
    rmse = np.sqrt(mse)
    
    # NASA Score
    # s = exp(d/13) - 1 if d < 0
    # s = exp(d/10) - 1 if d >= 0
    # d = pred - true
    d = pred_rul - true_rul
    score = 0
    for diff in d:
        if diff < 0:
            score += np.exp(-diff/13) - 1
        else:
            score += np.exp(diff/10) - 1
            
    # R2 Score
    r2 = r2_score(true_rul, pred_rul)
            
    return rmse, score, r2

def evaluate_model():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Data
    data_raw_path = config['paths']['data_raw']
    subset = config['data']['dataset']
    
    train_df, test_df, rul_df = load_data(data_raw_path, subset)
    
    # Process Data
    # IMPORTANT: We must use the scaler fitted on TRAIN data to transform TEST data
    # In `process_data`, we did fit_transform on train and transform on test.
    # We need to repeat exactly that process to ensure consistency.
    # To do this correctly without reloading train, we should have saved the scaler. 
    # But for now, we reload train and re-fit scaler as established in process_data.
    _, test_df, feature_cols = process_data(train_df, test_df, subset)
    
    # Dataset for Test
    # For test, we only care about the last sequence of each unit to predict the final RUL
    seq_len = config['data']['sequence_length']
    test_dataset = CMAPSSDataset(test_df, feature_cols, seq_len, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Load Model
    model = RULPredictor(
        input_dim=len(feature_cols),
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers']
    ).to(device)
    
    checkpoint_path = os.path.join(config['paths']['checkpoints_dir'], "best_model.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    predictions = []
    unit_ids = []
    
    with torch.no_grad():
        for X, y in tqdm(test_loader, desc="Evaluating"):
            X = X.to(device)
            # y contains unit_id in test mode (as float tensor)
            out = model(X)
            predictions.append(out.item())
            unit_ids.append(int(y.item()))
            
    predictions = np.array(predictions)
    unit_ids = np.array(unit_ids)
    
    # True RUL
    # rul_df index 0 -> unit 1
    # We need to select RULs for the unit_ids we have
    # unit_ids are 1-based (from data). numpy array index is 0-based.
    
    # Filter True RUL
    # rul_df has shape (100, 1).
    final_true_rul = []
    for uid in unit_ids:
        # uid is 1-based, accessing index uid-1
        final_true_rul.append(rul_df.iloc[uid-1]['RUL'])
        
    true_rul = np.array(final_true_rul)
    
    # Ensure lengths match
    # CMAPSS test files sometimes have units with length < sequence_length.
    # Our dataset skips those. We must align true_rul.
    # Check which units made it into the dataset.
    # In `dataset.py`, we iterate units 1..N. If unit length < seq_len, it's skipped.
    # We should track valid units.
    # Re-checking dataset logic: it iterates by groupby(unit).
    # We need to know which units were skipped.
    
    # Quick fix: In dataset mode='test', returning unit_id would help alignment.
    # But for now, assuming FD001 units are all long enough (usually > 30).
    # Only 100 predictions expected.
    
    if len(predictions) != len(true_rul):
        logger.warning(f"Mismatch remained: P:{len(predictions)} vs T:{len(true_rul)}")

    rmse, score, r2 = compute_metrics(true_rul, predictions)
    
    logger.info(f"\nTest RMSE: {rmse:.4f}")
    logger.info(f"Test NASA Score: {score:.4f}")
    logger.info(f"Test R2 Score: {r2:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(true_rul, label='True RUL')
    plt.plot(predictions, label='Predicted RUL')
    plt.legend()
    plt.title(f"Test Set Predictions (RMSE={rmse:.2f}, R2={r2:.2f})")
    plt.savefig(os.path.join(config['paths']['output_dir'], "test_predictions.png"))
    logger.info("Saved prediction plot.")

if __name__ == "__main__":
    evaluate_model()
