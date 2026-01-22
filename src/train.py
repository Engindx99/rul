import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from src.data.loader import load_data
from src.data.preprocess import process_data, add_rul
from src.data.dataset import CMAPSSDataset
from src.models.lstm import RULPredictor
from src.utils.logger import get_logger

logger = get_logger("Train")

def load_config(config_path="config/config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train_model():
    config = load_config()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Paths
    data_raw_path = config['paths']['data_raw']
    subset = config['data']['dataset']
    
    # 1. Load Data
    logger.info("Loading Data...")
    train_df, test_df, rul_df = load_data(data_raw_path, subset)
    
    # 2. Add RUL to Train
    train_df = add_rul(train_df, max_rul=125)
    
    # 3. Process (Scale, Drop sensors)
    logger.info("Processing Data...")
    train_df, test_df, feature_cols = process_data(train_df, test_df, subset)
    logger.info(f"Using Features: {feature_cols}")
    
    # 4. Create Datasets
    seq_len = config['data']['sequence_length']
    batch_size = config['data']['batch_size']
    
    # Split train for validation (simple unit-based split to avoid leakage)
    units = train_df['unit'].unique()
    val_split = config['data']['validation_split']
    n_val = int(len(units) * val_split)
    train_units = units[:-n_val]
    val_units = units[-n_val:]
    
    train_data = train_df[train_df['unit'].isin(train_units)]
    val_data = train_df[train_df['unit'].isin(val_units)]
    
    train_dataset = CMAPSSDataset(train_data, feature_cols, seq_len, mode='train')
    val_dataset = CMAPSSDataset(val_data, feature_cols, seq_len, mode='train')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # 5. Model
    model = RULPredictor(
        input_dim=len(feature_cols),
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()
    
    epochs = config['training']['epochs']
    best_val_loss = float('inf')
    
    train_losses = []
    val_losses = []

    # 6. Train Loop
    logger.info("Starting Training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs.view(-1), y) # Flatten outputs to match y
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['clip_grad'])
            optimizer.step()
            
            running_loss += loss.item() * X.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs.view(-1), y)
                val_running_loss += loss.item() * X.size(0)
        
        val_loss = val_running_loss / len(val_dataset)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        logger.info(f"Epoch {epoch+1} - Train RMSE: {np.sqrt(epoch_loss):.4f} - Val RMSE: {np.sqrt(val_loss):.4f}")
        
        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(config['paths']['checkpoints_dir'], "best_model.pth"))
            logger.info("Saved Best Model")

    # Plot Loss
    plt.figure()
    plt.plot(train_losses, label='Train Loss (MSE)')
    plt.plot(val_losses, label='Val Loss (MSE)')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(config['paths']['output_dir'], "loss_curve.png"))
    logger.info("Training Complete.")

if __name__ == "__main__":
    train_model()
