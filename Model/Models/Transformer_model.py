import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import fbeta_score
import optuna  # 🚀 引入自動調參套件

from base_model import BasePM25Model, TARGET_COL, EXCEEDANCE_THRESHOLD

# -------------------------
# 1. 位置編碼器
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x

# -------------------------
# 2. 原生 PyTorch Transformer 分類器
# -------------------------
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.feature_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.feature_proj(x)                 
        x = self.pos_encoder(x)                  
        x = self.transformer_encoder(x)          
        last_time_step_out = x[:, -1, :]         
        prob = self.classifier(last_time_step_out)
        return prob.squeeze(-1)

# -------------------------
# 3. 繼承封裝類別 (含 Optuna 調參機制)
# -------------------------
class VanillaTransformerModel(BasePM25Model):
    def __init__(self, candidate_features=None, target_col=TARGET_COL, threshold=EXCEEDANCE_THRESHOLD, seq_len=48, horizon=3):
        super().__init__(candidate_features, target_col, threshold, horizon)
        self.seq_len = seq_len
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_cols = []
        self.best_params = {} # 儲存調參結果
        
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

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None):
        # ==========================================
        # 1. 處理 Train Data (2023 以前)
        # ==========================================
        processed_train = self.preprocess(train_df)
        labels_train = (processed_train[self.target_col].values > self.threshold).astype(int)
        
        # 建立 Scaler 並轉換訓練集
        scaled_train = self.scaler.fit_transform(processed_train.values)
        X_train, y_train = self._create_sequences(scaled_train, labels_train)
        
        # 計算不平衡權重 (針對高 Recall)
        pos_ratio = (y_train == 0).sum() / max(1, (y_train == 1).sum())
        pos_weight = torch.tensor([pos_ratio], dtype=torch.float32).to(self.device)

        # ==========================================
        # 2. Optuna 調參 (使用你定義的 2024 val_data)
        # ==========================================
        if val_df is not None:
            print("      [Tuning] Using User-Defined Validation Set (2024) for Optuna...")
            
            # 處理 Validation Data (必須使用 train 的 scaler)
            processed_val = self.preprocess(val_df)
            labels_val = (processed_val[self.target_col].values > self.threshold).astype(int)
            scaled_val = self.scaler.transform(processed_val.values)
            X_val, y_val = self._create_sequences(scaled_val, labels_val)
            
            # 將訓練集包裝給 Optuna 內的試驗模型
            tune_train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
            
            def objective(trial):
                d_model = trial.suggest_categorical('d_model', [32, 64]) 
                nhead = trial.suggest_categorical('nhead', [2, 4])
                num_layers = trial.suggest_int('num_layers', 1, 3)
                lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
                dropout = trial.suggest_float('dropout', 0.1, 0.4)
                
                temp_model = TimeSeriesTransformer(
                    input_dim=len(self.feature_cols), d_model=d_model, nhead=nhead, 
                    num_layers=num_layers, dropout=dropout
                ).to(self.device)
                
                optimizer = torch.optim.Adam(temp_model.parameters(), lr=lr)
                criterion = nn.BCELoss(weight=pos_weight)
                
                temp_model.train()
                for epoch in range(5):  # 快速試驗 5 個 epoch
                    for bx, by in tune_train_loader:
                        bx, by = bx.to(self.device), by.to(self.device)
                        optimizer.zero_grad()
                        preds = temp_model(bx)
                        loss = criterion(preds, by)
                        loss.backward()
                        optimizer.step()
                
                # 拿訓練好的試驗模型去預測 val_df
                temp_model.eval()
                with torch.no_grad():
                    val_preds_prob = temp_model(X_val.to(self.device)).cpu().numpy()
                
                val_preds = (val_preds_prob > 0.5).astype(int)
                score = fbeta_score(y_val.numpy(), val_preds, beta=2, zero_division=0)
                return score

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=10) # 測試 10 種架構
            
            self.best_params = study.best_params
            print(f"      [Tuning] Best Params Found: {self.best_params}")
        else:
            # 如果沒傳入 val_df 的防呆機制
            print("      [Tuning] No validation set provided. Using default architecture.")
            self.best_params = {'d_model': 64, 'nhead': 4, 'num_layers': 2, 'lr': 0.001, 'dropout': 0.2}

        # ==========================================
        # 3. 正式訓練 (使用最佳參數訓練 2023 以前的所有資料)
        # ==========================================
        self.model = TimeSeriesTransformer(
            input_dim=len(self.feature_cols), 
            d_model=self.best_params['d_model'], 
            nhead=self.best_params['nhead'], 
            num_layers=self.best_params['num_layers'], 
            dropout=self.best_params['dropout']
        ).to(self.device)
        
        full_train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
        final_criterion = nn.BCELoss(weight=pos_weight)
        final_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.best_params['lr'])
        
        self.model.train()
        torch.manual_seed(2026)
        
        # 正式訓練跑 15 個 epoch
        for epoch in range(15):
            for batch_x, batch_y in full_train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                final_optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = final_criterion(preds, batch_y)
                loss.backward()
                final_optimizer.step()
                
        self.is_fitted = True
        self.selected_features = self.candidate_features

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
            
        processed_df = self.preprocess(df)
        scaled_data = self.scaler.transform(processed_df.values)
        dummy_labels = np.zeros(len(scaled_data))
        
        X_test, _ = self._create_sequences(scaled_data, dummy_labels)
        
        self.model.eval()
        with torch.no_grad():
            X_test = X_test.to(self.device)
            preds = self.model(X_test)
            
        probs = preds.cpu().numpy()
        
        # 對齊預測盲區
        pad_len = self.seq_len + self.horizon - 1
        padded_probs = np.pad(probs, (pad_len, 0), 'constant', constant_values=0)
        return padded_probs