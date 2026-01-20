import torch
import torch.nn as nn

class RULPredictorLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1):
        super(RULPredictorLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM Katmanı
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        
        # Tam bağlantılı (Fully Connected) katmanlar
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        # h0 ve c0 başlangıç durumları (default olarak sıfır)
        out, _ = self.lstm(x)
        
        # Sadece son zaman adımının (last time step) çıktısını alıyoruz
        out = out[:, -1, :]
        
        # RUL tahmini için FC katmanına gönderiyoruz
        out = self.fc(out)
        return out