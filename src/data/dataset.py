import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class CMAPSSDataset(Dataset):
    def __init__(self, data: pd.DataFrame, feature_cols: list, sequence_length: int, mode='train'):
        """
        Args:
            data: DataFrame with features and RUL.
            feature_cols: List of feature column names.
            sequence_length: Window size for LSTM.
            mode: 'train' or 'test'. 
                  In 'test' mode (for evaluation on test set), we might only take the *last* sequence per unit or all.
                  Standard RUL evaluation is often on the LAST time step of the test unit.
                  However, for 'validation' split from train, we treat it like train.
        """
        self.data = data
        self.feature_cols = feature_cols
        self.sequence_length = sequence_length
        self.mode = mode
        
        # Generate sequences
        self.samples = []
        self.targets = []
        
        self._generate_sequences()

    def _generate_sequences(self):
        # Group by unit
        grouped = self.data.groupby('unit')
        
        for unit_id, group in grouped:
            group_values = group[self.feature_cols].values
            # RUL is target
            # Use 'RUL' column if it exists (it should)
            if 'RUL' in group.columns:
                rul_values = group['RUL'].values
            else:
                rul_values = None # Should not happen in this design

            num_rows = group_values.shape[0]
            
            if self.mode == 'train':
                # Sliding window
                for start in range(num_rows - self.sequence_length + 1):
                    end = start + self.sequence_length
                    seq = group_values[start:end, :]
                    target = rul_values[end-1] # Target is RUL at the last step of the window
                    
                    self.samples.append(seq)
                    self.targets.append(target)
            
            elif self.mode == 'test':
                # For standard CMAPSS test evaluation, we only predict RUL at the LAST time step
                # The provided RUL_FD001.txt is the RUL at the end of the test sequence.
                # So we take the LAST sequence of length N from the test data.
                if num_rows >= self.sequence_length:
                    seq = group_values[num_rows - self.sequence_length : num_rows, :]
                    self.samples.append(seq)
                    # For test set, the target is in a separate DF usually, we handle it outside or join it.
                    # Here we might just store unit_id to match later.
                    self.targets.append(unit_id) 

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq = self.samples[idx]
        target = self.targets[idx]
        
        # In test mode, we might need unit_id to align with RUL file
        # But 'target' stores unit_id in test mode (see _generate_sequences)
        # However, target_tensor casts it to float.
        # Let's verify what self.targets stores.
        # In train: target = RUL value
        # In test: target = unit_id
        
        seq_tensor = torch.tensor(seq, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32) 
        
        return seq_tensor, target_tensor
