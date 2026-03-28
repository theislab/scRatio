import sys
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from model import ProbabilisticClassifier

def train_classifier(X_train, y_train, model):
    # Create a DataLoader for the training data
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

    # Define the optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    model.train()
    for epoch in range(200):  # Number of epochs
        total_loss = 0
        for X_batch, y_batch in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")

    return model

@hydra.main(config_path="./config", config_name="train", version_base=None)
def main(config: DictConfig):    
    # Initialize device 
    device = "cuda" if torch.cuda.is_available() else "cpu"
   
    # Read data
    data_path = Path(config.paths.data_path)
    dimensions = config.paths.dimensions
    
    X_block_train = np.load(data_path / f"block_sigma_{dimensions}_train.npy")
    X_block_val = np.load(data_path / f"block_sigma_{dimensions}_val.npy")
    X_block_test = np.load(data_path / f"block_sigma_{dimensions}_test.npy")
    
    X_prior_train = np.load(data_path / f"identity_sigma_{dimensions}_train.npy")
    X_prior_val = np.load(data_path / f"identity_sigma_{dimensions}_val.npy")
    X_prior_test = np.load(data_path / f"identity_sigma_{dimensions}_test.npy")
    
    # Get training and test sets
    X_block_train, X_prior_train =  torch.from_numpy(X_block_train).to(device), torch.from_numpy(X_prior_train).to(device)
    X_block_val, X_prior_val = torch.from_numpy(X_block_val).to(device), torch.from_numpy(X_prior_val).to(device)
    X_block_test, X_prior_test = torch.from_numpy(X_block_test).to(device), torch.from_numpy(X_prior_test).to(device)
    
    X_train = torch.cat([X_block_train, X_prior_train], dim=0)
    X_val = torch.cat([X_block_val, X_prior_val], dim=0)
    X_test = torch.cat([X_block_test, X_prior_test], dim=0)
    
    y_train = torch.cat([torch.ones(X_block_train.shape[0]), torch.zeros(X_prior_train.shape[0])], dim=0).long().to(device)
    y_val = torch.cat([torch.ones(X_block_val.shape[0]), torch.zeros(X_prior_val.shape[0])], dim=0).long().to(device)
    y_test = torch.cat([torch.ones(X_block_test.shape[0]), torch.zeros(X_prior_test.shape[0])], dim=0).long().to(device)
    
    input_dim = X_train.shape[1]
    output_dim = 2  # Binary classification
    
    # Resiult directory
    for iteration in range(3):
        # Initialize target folders for results
        res_dir = Path(config.paths.res_dir)  
        res_dir = res_dir / f"dimensions_{input_dim}_hidden_dim_{config.model.hidden_dim}_hidden_layers_no_{config.model.num_hidden_layers}_iteration_{iteration}"
        res_dir.mkdir(parents=True, exist_ok=True)  
    
        # Initialize the model 
        model = ProbabilisticClassifier(input_dim, 
                                        config.model.hidden_dim, 
                                        output_dim,
                                        config.model.num_hidden_layers).to(device)
    
        # Train the model
        metrics = {"accuracy_val": None, "accuracy_test": None}
        model = train_classifier(X_train, y_train, model)
        model.eval()
        with torch.no_grad():
            accuracy_val = (model(X_val).argmax(dim=1) == y_val).float().mean().item()
            accuracy_test = (model(X_test).argmax(dim=1) == y_test).float().mean().item()
            
        metrics["accuracy_val"] = accuracy_val
        metrics["accuracy_test"] = accuracy_test
    
        # Compute ratios 
        with torch.no_grad():
            log_ratios_val = model.compute_log_ratio(X_block_val, y_idx=1, y_prime_idx=0).cpu().numpy()
            log_ratios_test = model.compute_log_ratio(X_block_test, y_idx=1, y_prime_idx=0).cpu().numpy()
        np.save(res_dir / "log_ratios_val.npy", log_ratios_val)
        np.save(res_dir / "log_ratios_test.npy", log_ratios_test)
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(res_dir / "metrics.csv", index=False)

if __name__ == "__main__":
    # running the experiment
    try: 
        main()
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
