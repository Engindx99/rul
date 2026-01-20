import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

class CMAPSSDataset(Dataset):
    def __init__(self, file_path, window_size=30, is_test=False, scaler=None, max_rul=125):
        self.window_size = window_size
        self.is_test = is_test
        self.max_rul = max_rul
        self.feature_cols = ['setting_1', 'setting_2', 'setting_3'] + ['s_{}'.format(i) for i in range(1, 22)]
        
        # Sütun isimleri
        column_names = ['unit_number', 'time_cycles', 'setting_1', 'setting_2', 'setting_3'] + \
                       ['s_{}'.format(i) for i in range(1, 22)]
        
        # Veriyi yükle
        df = pd.read_csv(file_path, sep='\s+', header=None, names=column_names)
        
        if not self.is_test:
            df = self._add_rul(df)
        
        # Normalizasyon
        if scaler is None:
            self.scaler = MinMaxScaler()
            df[self.feature_cols] = self.scaler.fit_transform(df[self.feature_cols])
        else:
            self.scaler = scaler
            df[self.feature_cols] = self.scaler.transform(df[self.feature_cols])
            
        self.sequences, self.labels = self._create_sequences(df)
        
    def _add_rul(self, df):
        max_cycle = df.groupby('unit_number')['time_cycles'].transform('max')
        df['RUL'] = (max_cycle - df['time_cycles']).clip(upper=self.max_rul)
        return df

    def _create_sequences(self, df):
        sequences, labels = [], []
        for unit in df['unit_number'].unique():
            unit_df = df[df['unit_number'] == unit]
            if len(unit_df) < self.window_size: continue
            
            data = unit_df[self.feature_cols].values
            if not self.is_test:
                target = unit_df['RUL'].values
                for i in range(len(unit_df) - self.window_size + 1):
                    sequences.append(data[i:i+self.window_size])
                    labels.append(target[i+self.window_size-1])
            else:
                sequences.append(data[-self.window_size:])
        return np.array(sequences), np.array(labels)

    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx):
        if not self.is_test:
            return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)
        return torch.tensor(self.sequences[idx], dtype=torch.float32)