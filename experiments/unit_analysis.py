import torch
import numpy as np
import joblib
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

# Root directory connection
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.model import RULPredictorLSTM

def run_experiment_plot(unit_ids=[11, 24, 50]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Path configurations
    scaler_path = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
    model_path = os.path.join(BASE_DIR, 'models', 'lstm_model.pth')
    data_path = os.path.join(BASE_DIR, 'data', 'train_FD001.txt')

    # Load artifacts
    scaler = joblib.load(scaler_path)
    model = RULPredictorLSTM(input_dim=24, hidden_dim=128, num_layers=2).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load data
    df = pd.read_csv(data_path, sep='\s+', header=None)
    feature_cols = df.columns[2:26]

    for unit_id in unit_ids:
        unit_data = df[df[0] == unit_id]
        total_cycles = len(unit_data)
        
        # Ground Truth RUL (Piecewise 125)
        actual_rul = np.arange(total_cycles)[::-1]
        actual_rul_clipped = np.clip(actual_rul, 0, 125)
        
        # Model Prediction
        scaled_features = scaler.transform(unit_data[feature_cols])
        unit_preds = []
        
        with torch.no_grad():
            for t in range(30, total_cycles):
                seq = torch.tensor(scaled_features[t-30:t], dtype=torch.float32).to(device).unsqueeze(0)
                pred = model(seq).item()
                unit_preds.append(pred)

        # Prepare metrics
        y_true = actual_rul_clipped[30:]
        y_pred = np.array(unit_preds)
        
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # Visualization
        plt.figure(figsize=(10, 5))
        plt.plot(y_true, label='True RUL', color='blue', linewidth=2)
        plt.plot(y_pred, label='Predicted RUL', color='red')
        
        # Metrics text (No box, No bold)
        metrics_text = f'RMSE: {rmse:.2f}\n$R^2$: {r2:.4f}'
        plt.gca().text(0.05, 0.05, metrics_text, 
                       transform=plt.gca().transAxes, 
                       fontsize=10, 
                       color='black')

        plt.title(f'Unit #{unit_id}')
        plt.xlabel('Cycle')
        plt.ylabel('RUL')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

if __name__ == "__main__":
    run_experiment_plot()