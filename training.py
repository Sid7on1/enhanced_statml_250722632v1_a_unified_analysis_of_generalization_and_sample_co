import logging
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants and configuration
CONFIG = {
    'model': 'L-Layer',
    'activation': 'ReLU',
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 10,
    'log_interval': 100,
    'save_interval': 1000,
    'data_path': 'data.csv',
    'model_path': 'model.pth'
}

class EnhancedStatDataset(Dataset):
    def __init__(self, data_path: str):
        self.data = pd.read_csv(data_path)
        self.columns = self.data.columns

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        sample = self.data.iloc[idx]
        return {
            'input': torch.tensor(sample[self.columns[:-1]].values, dtype=torch.float32),
            'target': torch.tensor(sample[self.columns[-1]].values, dtype=torch.float32)
        }

class LLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(LLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class DomainAdversarial(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(DomainAdversarial, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class TrainingPipeline:
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _create_model(self):
        if self.config['model'] == 'L-Layer':
            self.model = LLayer(input_dim=784, hidden_dim=256, output_dim=10)
        elif self.config['model'] == 'Domain-Adversarial':
            self.model = DomainAdversarial(input_dim=784, hidden_dim=256, output_dim=10)
        else:
            raise ValueError('Invalid model type')

        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.criterion = nn.MSELoss()

    def _train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(self.data_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            if batch_idx % self.config['log_interval'] == 0:
                logger.info(f'Train Epoch: {epoch+1} [{batch_idx*len(inputs)}/{len(self.data_loader.dataset)}] Loss: {total_loss/(batch_idx+1):.4f}')
        logger.info(f'Train Epoch: {epoch+1} Loss: {total_loss/(batch_idx+1):.4f}')

    def _validate_epoch(self, epoch: int):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in self.data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        logger.info(f'Validation Epoch: {epoch+1} Loss: {total_loss/(len(self.data_loader.dataset)):.4f}')

    def train(self):
        self._create_model()
        self.data_loader = DataLoader(EnhancedStatDataset(CONFIG['data_path']), batch_size=CONFIG['batch_size'], shuffle=True)
        for epoch in range(CONFIG['epochs']):
            self._train_epoch(epoch)
            self._validate_epoch(epoch)
            if epoch % CONFIG['save_interval'] == 0:
                torch.save(self.model.state_dict(), CONFIG['model_path'])
        torch.save(self.model.state_dict(), CONFIG['model_path'])

if __name__ == '__main__':
    start_time = time.time()
    training_pipeline = TrainingPipeline(CONFIG)
    training_pipeline.train()
    logger.info(f'Training completed in {time.time()-start_time:.2f} seconds')