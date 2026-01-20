import torch
import numpy as np
import joblib
import os
import sys
import matplotlib.pyplot as plt

# Proje kök dizinini ekle
sys.path.append(os.getcwd())

from src.dataset import CMAPSSDataset
from src.model import RULPredictorLSTM

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Kayıtlı Varlıkları Yükle
    scaler = joblib.load('models/scaler.pkl')
    model = RULPredictorLSTM(input_dim=24, hidden_dim=128, num_layers=2).to(device)
    model.load_state_dict(torch.load('models/lstm_model.pth'))
    model.eval()

    # 2. Test Verisini Hazırla
    test_path = os.path.join('data', 'test_FD001.txt')
    # Test setinde her motorun sadece SON penceresini alıyoruz (RUL tahmini için)
    test_dataset = CMAPSSDataset(test_path, window_size=30, is_test=True, scaler=scaler)
    
    # Gerçek RUL değerlerini oku
    true_rul_path = os.path.join('data', 'RUL_FD001.txt')
    true_ruls = np.loadtxt(true_rul_path)

    # 3. Tahmin Yap
    predictions = []
    with torch.no_grad():
        for i in range(len(test_dataset)):
            seq = test_dataset[i].to(device).unsqueeze(0) # Batch boyutu ekle
            output = model(seq)
            predictions.append(output.item())

    predictions = np.array(predictions)
    
    # NASA'nın Piecewise RUL mantığına göre gerçek değerleri de kırpabiliriz (opsiyonel)
    # true_ruls = np.clip(true_ruls, 0, 125)

    # 4. Metrikler
    rmse = np.sqrt(np.mean((predictions - true_ruls)**2))
    print(f"\nTest Sonuçları (FD001):")
    print(f"RMSE: {rmse:.2f}")

    # 5. Görselleştirme
    plt.figure(figsize=(12, 6))
    plt.plot(true_ruls, label='Gerçek RUL', color='blue', marker='o', markersize=3)
    plt.plot(predictions, label='Tahmin RUL', color='red', linestyle='--')
    plt.title('FD001 - Gerçek vs Tahmin RUL')
    plt.xlabel('Motor Numarası (Unit)')
    plt.ylabel('Kalan Ömür (Cycle)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    evaluate()