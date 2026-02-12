import os
import torch
import pandas as pd
import numpy as np

class GenomicDataset(torch.utils.data.Dataset):
    def __init__(self, data=None, labels=None, data_file=None):
        if data is None and labels is None:
            df = pd.read_csv(data_file, index_col=0)
            # Extract labels (pseudo_id and label columns)
            if "pseudo_id" in df.columns:
                df = df.drop('pseudo_id', axis=1)
            labels_df = df[[ 'label']]
            # Extract data (all columns including pseudo_id, except label)
            data_df = df.drop('label', axis=1)
            # Convert to float32 to match model weights
            self.data = data_df.to_numpy().astype(np.float32)
            self.labels = labels_df.to_numpy().astype(np.float32)
        else:
            self.data = data.cpu().numpy().astype(np.float32)
            self.labels = labels.cpu().numpy().astype(np.float32)

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
        
if __name__ == "__main__":

    SEED = 0
    ALPHA = 0.8

    data_csv = os.path.join("/workspace", "DEMO", "genomic_regressor", "data", "CNAE-9-wide.csv")
    labels_csv = os.path.join("/workspace", "DEMO", "genomic_regressor", "data", "CNAE-9-labels.csv")

    # Read files
    feutures_df = pd.read_csv(data_csv, index_col=0) 
    labels_df = pd.read_csv(labels_csv, index_col=0) 

    # Extract all feutures and labels for the train/valid splitting
    features = features_df.values
    labels = labels_df.values.flatten()

    # Convert to Tensors
    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    # Shuffle
    torch.manual_seed(SEED)
    num_samples = features_tensor.shape[0]
    indices = torch.randperm(num_samples)

    # Apply shuffle
    shuffled_features = features_tensor[indices]
    shuffled_labels = labels_tensor[indices]

    # Split SIZE
    train_size = int(ALPHA * num_samples)
    #valid_size = num_samples - train_size

    # Split Data
    X_train = shuffled_features[:train_size]
    y_train = shuffled_labels[:train_size]
    X_valid = shuffled_features[train_size:]
    y_valid = shuffled_labels[train_size:]

    # Create TensorDataset
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    valid_dataset = torch.utils.data.TensorDataset(X_valid, y_valid)

    # Create DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False)
