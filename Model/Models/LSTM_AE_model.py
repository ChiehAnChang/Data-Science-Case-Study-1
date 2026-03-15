import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from base_model import BasePM25Model, TARGET_COL, EXCEEDANCE_THRESHOLD

class LSTMAutoencoderClassifier(nn.Module):
    """
    A PyTorch Neural Network combining an LSTM Autoencoder with a downstream Classifier.
    The Autoencoder extracts a robust latent representation of sequential time-series data,
    while the Classifier branch maps this latent vector to a binary exceedance probability.
    """
    def __init__(self, input_dim, hidden_dim, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # Encoder mapping input sequence to a latent context vector
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        # Decoder mapping latent context vector back to the original sequence for reconstruction
        self.decoder_lstm = nn.LSTM(hidden_dim, input_dim, batch_first=True)
        
        # Multi-layer perceptron generating probability scores from the latent vector
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass executing both the reconstruction branch and classification branch.
        """
        _, (hidden, _) = self.encoder_lstm(x)
        latent_vector = hidden[-1]  # Get the final hidden state of the LSTM
        
        # Repeat the latent vector to match the sequence length required for decoding
        latent_repeated = latent_vector.unsqueeze(1).repeat(1, self.seq_len, 1)
        reconstructed_x, _ = self.decoder_lstm(latent_repeated)
        
        # Predict the probability using the latent context
        prob = self.classifier(latent_vector)
        return reconstructed_x, prob.squeeze(-1)

class AutoencoderLSTMModel(BasePM25Model):
    """
    Model wrapper for the PyTorch LSTM Autoencoder Classifier enforcing conformity
    to the BasePM25Model interface. Manages data scaling, windowing (sequence generation), and device mapping.
    """
    def __init__(self, candidate_features=None, target_col=TARGET_COL, threshold=EXCEEDANCE_THRESHOLD, seq_len=48, horizon=3):
        super().__init__(candidate_features, target_col, threshold, horizon)
        self.seq_len = seq_len # length of the historical window to look at (e.g., 48 hours)
        self.model = None
        self.scaler = StandardScaler() # Standardize features to improve gradient convergence
        
        # Dynamically determine the best compute hardware available (Apple Silicon MPS, CUDA, or CPU)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.feature_cols = []
        
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Forward-fills missing data to prevent holes in sequential data processing 
        and extracts localized temporal features.
        """
        processed_df = df.copy()
        processed_df['hour'] = processed_df.index.hour
        processed_df['dayofweek'] = processed_df.index.dayofweek
        processed_df['month'] = processed_df.index.month
        
        for col in self.candidate_features:
            processed_df[col] = processed_df.get(col, 0.0).fillna(0).astype(float)
            
        self.feature_cols = [self.target_col, 'hour', 'dayofweek', 'month'] + self.candidate_features
        # Ensure sequential continuity using forward fill (ffill) instead of dropna mapping
        return processed_df[self.feature_cols].ffill().fillna(0)

    def _create_sequences(self, data: np.ndarray, labels: np.ndarray):
        """
        Transforms 2D tabular data into 3D tensors shaped as (samples, sequence_length, features)
        suitable for recurrent neural networks processing continuous time segments.
        
        Args:
            data (np.ndarray): Scaled feature matrix.
            labels (np.ndarray): Target binary labels.
        """
        X, y = [], []
        # Generate rolling windows of data shifted by the horizon
        for i in range(len(data) - self.seq_len - self.horizon + 1):
            X.append(data[i : i + self.seq_len])
            y.append(labels[i + self.seq_len + self.horizon - 1]) 
        return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)

    def fit(self, train_df: pd.DataFrame):
        """
        Trains the autoencoder-LSTM model using a joint loss function optimizing for
        both sequence reconstruction (MSE) and classification accuracy (weighted BCE).
        """
        processed_train = self.preprocess(train_df)
        labels = (processed_train[self.target_col].values > self.threshold).astype(int)
        
        # Scale inputs 
        scaled_data = self.scaler.fit_transform(processed_train.values)
        
        # Convert to 3D PyTorch tensors and chunk into standard DataLoaders
        X_train, y_train = self._create_sequences(scaled_data, labels)
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        self.model = LSTMAutoencoderClassifier(len(self.feature_cols), 32, self.seq_len).to(self.device)
        
        # Loss functions mapping: MSE for signal reconstruction
        criterion_recon = nn.MSELoss()
        
        # Handling the positive class imbalance mathematically using custom tensor weights maps
        pos_ratio = (labels == 0).sum() / max(1, (labels == 1).sum())
        pos_ratio = torch.tensor(pos_ratio, dtype=torch.float32).to(self.device)
        criterion_class = nn.BCELoss(reduction='none') 
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.train()
        torch.manual_seed(2026)
        for epoch in range(15):
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                reconstructed_x, preds = self.model(batch_x)
                
                # Reconstruction penalty ensuring the latent space is structurally sound
                recon_loss = criterion_recon(reconstructed_x, batch_x)
                
                # Classification penalty dynamically prioritizing the rare positive exceedance instances
                bce_loss = criterion_class(preds, batch_y)
                weight = torch.where(batch_y == 1, pos_ratio, 1.0)
                class_loss = (bce_loss * weight).mean()
                
                # Joint Backpropagation weighting classification twice as heavily as reconstruction
                loss = recon_loss + (2.0 * class_loss) 
                loss.backward()
                optimizer.step()
        self.is_fitted = True

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Applies window generation onto unseen input sequences to output aligned binary classification probability metrics.
        """
        if not self.is_fitted: raise ValueError("Model must be fitted before calling predict_proba.")
        processed_df = self.preprocess(df)
        scaled_data = self.scaler.transform(processed_df.values)
        
        # Predict dummy values and discard labels for structural mapping evaluations
        X_test, _ = self._create_sequences(scaled_data, np.zeros(len(scaled_data)))
        
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_test.to(self.device))[1]
            
        # Pad the initial periods lacking enough historical length context with zero probabilities
        # to ensure the array length perfectly matches the preprocessed DataFrame temporal index structure.
        pad_len = self.seq_len + self.horizon - 1
        return np.pad(preds.cpu().numpy(), (pad_len, 0), 'constant', constant_values=0)