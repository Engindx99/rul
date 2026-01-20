import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import joblib
import os
import sys

# Yol ayarı: src içindeyken bir üst dizini tanıması için
sys.path.append(os.getcwd())

from src.dataset import CMAPSSDataset
from src.model import RULPredictorLSTM

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs('models', exist_ok=True)
    
    # PARAMETRELER
    WINDOW_SIZE = 30
    EPOCHS = 30
    BATCH_SIZE = 64

    # VERİ BAĞLANTISI (Doğru yol burası)
    train_path = os.path.join('data', 'train_FD001.txt')
    print(f"Veri okunuyor: {train_path}")
    
    train_dataset = CMAPSSDataset(train_path, window_size=WINDOW_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = RULPredictorLSTM(input_dim=24, hidden_dim=128, num_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print("Eğitim başladı...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for seq, label in train_loader:
            seq, label = seq.to(device), label.to(device)
            out = model(seq)
            loss = criterion(out.squeeze(), label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

    # KAYIT
    torch.save(model.state_dict(), 'models/lstm_model.pth')
    joblib.dump(train_dataset.scaler, 'models/scaler.pkl')
    print("Model ve Scaler kaydedildi.")

if __name__ == "__main__":
    train()