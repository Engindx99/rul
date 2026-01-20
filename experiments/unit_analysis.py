import torch
import numpy as np
import joblib
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Experiments içinden bir üst dizine (root) erişmek için yolu bağla
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Artık src altındaki modülleri sorunsuz import edebiliriz
from src.model import RULPredictorLSTM

def run_experiment_plot(unit_ids=[1, 25, 50]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dosya yollarını ana dizine göre ayarla
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scaler_path = os.path.join(base_path, 'models', 'scaler.pkl')
    model_path = os.path.join(base_path, 'models', 'lstm_model.pth')
    data_path = os.path.join(base_path, 'data', 'train_FD001.txt')

    # 1. Yüklemeler
    scaler = joblib.load(scaler_path)
    model = RULPredictorLSTM(input_dim=24, hidden_dim=128, num_layers=2).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 2. Veri Okuma
    df = pd.read_csv(data_path, sep='\s+', header=None)
    feature_cols = df.columns[2:26]

    for unit_id in unit_ids:
        unit_data = df[df[0] == unit_id]
        total_cycles = len(unit_data)
        
        # Gerçek RUL
        actual_rul = np.arange(total_cycles)[::-1]
        actual_rul_clipped = np.clip(actual_rul, 0, 125)
        
        # Tahmin
        scaled_features = scaler.transform(unit_data[feature_cols])
        unit_preds = []
        
        with torch.no_grad():
            for t in range(30, total_cycles):
                seq = torch.tensor(scaled_features[t-30:t], dtype=torch.float32).to(device).unsqueeze(0)
                pred = model(seq).item()
                unit_preds.append(pred)

        # Görselleştirme
        plt.figure(figsize=(10, 5))
        plt.plot(actual_rul_clipped[30:], label='Gerçek RUL', color='blue', linewidth=2)
        plt.plot(unit_preds, label='Tahmin RUL', color='red', linestyle='--')
        plt.title(f'Deney Analizi: Motor #{unit_id}')
        plt.xlabel('Zaman (Cycle)')
        plt.ylabel('RUL')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

if __name__ == "__main__":
    run_experiment_plot(unit_ids=[5, 21, 82])