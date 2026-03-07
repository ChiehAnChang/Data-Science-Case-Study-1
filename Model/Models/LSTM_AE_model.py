import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from base_model import BasePM25Model, TARGET_COL, EXCEEDANCE_THRESHOLD

class LSTMAutoencoderClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder_lstm = nn.LSTM(hidden_dim, input_dim, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        _, (hidden, _) = self.encoder_lstm(x)
        latent_vector = hidden[-1] 
        latent_repeated = latent_vector.unsqueeze(1).repeat(1, self.seq_len, 1)
        reconstructed_x, _ = self.decoder_lstm(latent_repeated)
        prob = self.classifier(latent_vector)
        return reconstructed_x, prob.squeeze(-1)

class AutoencoderLSTMModel(BasePM25Model):
    def __init__(self, candidate_features=None, target_col=TARGET_COL, threshold=EXCEEDANCE_THRESHOLD, seq_len=48, horizon=3):
        super().__init__(candidate_features, target_col, threshold, horizon)
        self.seq_len = seq_len
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_cols = []
        
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        processed_df = df.copy()
        processed_df['hour'] = processed_df.index.hour
        processed_df['dayofweek'] = processed_df.index.dayofweek
        processed_df['month'] = processed_df.index.month
        for col in self.candidate_features:
            processed_df[col] = processed_df.get(col, 0.0).fillna(0).astype(float)
        self.feature_cols = [self.target_col, 'hour', 'dayofweek', 'month'] + self.candidate_features
        return processed_df[self.feature_cols].ffill().fillna(0)

    def _create_sequences(self, data: np.ndarray, labels: np.ndarray):
        X, y = [], []
        for i in range(len(data) - self.seq_len - self.horizon + 1):
            X.append(data[i : i + self.seq_len])
            y.append(labels[i + self.seq_len + self.horizon - 1]) 
        return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)

    def fit(self, train_df: pd.DataFrame):
        processed_train = self.preprocess(train_df)
        labels = (processed_train[self.target_col].values > self.threshold).astype(int)
        scaled_data = self.scaler.fit_transform(processed_train.values)
        X_train, y_train = self._create_sequences(scaled_data, labels)
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        self.model = LSTMAutoencoderClassifier(len(self.feature_cols), 32, self.seq_len).to(self.device)
        criterion_recon = nn.MSELoss()
        
        pos_ratio = (labels == 0).sum() / max(1, (labels == 1).sum())
        criterion_class = nn.BCELoss(weight=torch.tensor([pos_ratio], dtype=torch.float32).to(self.device)) 
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.train()
        torch.manual_seed(2026)
        for epoch in range(15):
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                reconstructed_x, preds = self.model(batch_x)
                loss = criterion_recon(reconstructed_x, batch_x) + (2.0 * criterion_class(preds, batch_y)) 
                loss.backward()
                optimizer.step()
        self.is_fitted = True

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted: raise ValueError("Model must be fitted.")
        processed_df = self.preprocess(df)
        scaled_data = self.scaler.transform(processed_df.values)
        X_test, _ = self._create_sequences(scaled_data, np.zeros(len(scaled_data)))
        
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_test.to(self.device))[1]
            
        pad_len = self.seq_len + self.horizon - 1
        return np.pad(preds.cpu().numpy(), (pad_len, 0), 'constant', constant_values=0)