import numpy as np
from sklearn.preprocessing import MinMaxScaler

class CMAPSSPreprocess:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.scaler = MinMaxScaler()
        self.sensor_cols = ['s_{}'.format(i) for i in range(1, 22)]
        self.setting_cols = ['setting_1', 'setting_2', 'setting_3']
        self.feature_cols = self.setting_cols + self.sensor_cols

    def fit_transform(self, df):
        # Sadece eğitim verisi üzerinde fit yapıyoruz
        df[self.feature_cols] = self.scaler.fit_transform(df[self.feature_cols])
        return df

    def transform(self, df):
        # Test verisi için fit yapmadan sadece dönüştürüyoruz (Data Leakage önlemi)
        df[self.feature_cols] = self.scaler.transform(df[self.feature_cols])
        return df

    def gen_sequence(self, df, seq_length, seq_cols):
        # Veriyi (pencere_sayısı, 30, feature_sayısı) formatına getirir
        data_matrix = df[seq_cols].values
        num_elements = data_matrix.shape[0]
        
        for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
            yield data_matrix[start:stop, :]

    def gen_labels(self, df, seq_length, label_cols):
        # Her pencereye karşılık gelen hedef RUL değerini üretir
        data_matrix = df[label_cols].values
        num_elements = data_matrix.shape[0]
        return data_matrix[seq_length:num_elements, :]