import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder

class CSVDataset(Dataset):
    def __init__(self, csv_path, target_col, normalize=True):
        df = pd.read_csv(csv_path)
        self.y = df[target_col].values
        self.X = df.drop(columns=[target_col])
        # кодирование 
        for col in self.X.select_dtypes(include=['object']):
            le = LabelEncoder()
            self.X[col] = le.fit_transform(self.X[col])
        if normalize:
            scaler = StandardScaler()
            self.X[:] = scaler.fit_transform(self.X)
        self.X = torch.tensor(self.X.values, dtype=torch.float32)
        self.y = torch.tensor(LabelEncoder().fit_transform(self.y), dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
