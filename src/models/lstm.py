import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(hidden_dim, 1)
        
    def forward(self, lstm_output):
        # lstm_output: (batch_size, seq_len, hidden_dim)
        
        # Energy: (batch_size, seq_len, 1)
        energy = self.attn(lstm_output)
        
        # Weights: (batch_size, seq_len, 1)
        weights = F.softmax(energy, dim=1)
        
        # Context structure: Sum(weights * lstm_output)
        # (batch_size, seq_len, 1) * (batch_size, seq_len, hidden_dim)
        # Result: (batch_size, hidden_dim)
        context = torch.sum(weights * lstm_output, dim=1)
        
        return context, weights

class RULPredictorObs(nn.Module):
    """
    Standard LSTM without Attention (Legacy)
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1, dropout=0.2):
        super(RULPredictorObs, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.regressor(out)

class RULPredictor(nn.Module):
    """
    LSTM with Attention Mechanism
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1, dropout=0.2):
        super(RULPredictor, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = Attention(hidden_dim)
        
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        # x: (batch, seq, input)
        lstm_out, _ = self.lstm(x)
        # lstm_out: (batch, seq, hidden)
        
        # Attention
        context_vector, attn_weights = self.attention(lstm_out)
        # context: (batch, hidden)
        
        # Head
        out = self.regressor(context_vector)
        
        return out
