import torch
import numpy as np
import joblib
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from src.model import RULPredictorLSTM

def plot_interactive(unit_ids=[10, 35, 49, 72, 85]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Kayıtlı Modeli ve Scaler'ı Yükle
    scaler = joblib.load('models/scaler.pkl')
    model = RULPredictorLSTM(input_dim=24, hidden_dim=128, num_layers=2).to(device)
    model.load_state_dict(torch.load('models/lstm_model.pth'))
    model.eval()

    # 2. Veriyi Oku
    train_path = os.path.join('data', 'train_FD001.txt')
    df = pd.read_csv(train_path, sep='\s+', header=None)
    feature_cols = df.columns[2:26]

    print(f"\n--- İnteraktif Mod Başlatıldı ---")
    print(f"Toplam {len(unit_ids)} motor incelenecek. Sıradaki grafiği görmek için mevcut pencereyi kapatın.\n")

    for unit_id in unit_ids:
        unit_data = df[df[0] == unit_id]
        total_cycles = len(unit_data)
        
        # Gerçek RUL ve Piecewise Clipping
        actual_rul = np.arange(total_cycles)[::-1]
        actual_rul_clipped = np.clip(actual_rul, 0, 125)
        
        # Tahmin Üretme
        scaled_features = scaler.transform(unit_data[feature_cols])
        unit_preds = []
        
        with torch.no_grad():
            for t in range(30, total_cycles):
                seq = torch.tensor(scaled_features[t-30:t], dtype=torch.float32).to(device).unsqueeze(0)
                pred = model(seq).item()
                unit_preds.append(pred)

        # Görselleştirme (Her Motor İçin Yeni Bir Figür)
        plt.figure(figsize=(10, 6))
        plt.plot(actual_rul_clipped[30:], label='Gerçek RUL (Hedef)', color='#1f77b4', linewidth=2)
        plt.plot(unit_preds, label='Model Tahmini', color='#d62728', linestyle='--', linewidth=2)
        
        plt.title(f'Motor Analiz Raporu: Unit #{unit_id}', fontsize=14)
        plt.xlabel('Cycle')
        plt.ylabel('Remaining Useful Life (RUL)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        print(f"Şu an gösteriliyor: Unit {unit_id} (Toplam Cycle: {total_cycles})")
        
        # plt.show() burada kodu durdurur. Pencereyi kapattığında döngü başa döner.
        plt.show()

if __name__ == "__main__":
    # İstediğin motor numaralarını buraya ekle
    plot_interactive(unit_ids=[1, 20, 45, 66, 99])