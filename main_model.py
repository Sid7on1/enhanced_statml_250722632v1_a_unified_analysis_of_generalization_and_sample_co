import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
from config import Config
from utils import load_data, save_model, load_model
from models import ReLUActivation, SoftplusActivation
from metrics import calculate_accuracy, calculate_precision, calculate_recall
from exceptions import ModelException, DataException
from constants import LR, SP

class MainModel(nn.Module):
    """
    Main computer vision model.
    """
    def __init__(self, config: Config):
        super(MainModel, self).__init__()
        self.config = config
        self.relu_activation = ReLUActivation()
        self.softplus_activation = SoftplusActivation()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor.
        """
        x = self.relu_activation(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu_activation(x)
        x = self.fc2(x)
        return x

    def train(self, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader):
        """
        Train the model.

        Args:
        train_loader (torch.utils.data.DataLoader): Training data loader.
        val_loader (torch.utils.data.DataLoader): Validation data loader.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.config.lr)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(self.config.epochs):
            self.train()
            total_loss = 0
            for batch in train_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.config.device), labels.to(self.config.device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            self.eval()
            total_correct = 0
            with torch.no_grad():
                for batch in val_loader:
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.config.device), labels.to(self.config.device)
                    outputs = self(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total_correct += (predicted == labels).sum().item()
            accuracy = total_correct / len(val_loader.dataset)
            logging.info(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}, Accuracy: {accuracy:.4f}')

    def evaluate(self, test_loader: torch.utils.data.DataLoader):
        """
        Evaluate the model.

        Args:
        test_loader (torch.utils.data.DataLoader): Test data loader.

        Returns:
        float: Model accuracy.
        """
        self.eval()
        total_correct = 0
        with torch.no_grad():
            for batch in test_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.config.device), labels.to(self.config.device)
                outputs = self(inputs)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
        accuracy = total_correct / len(test_loader.dataset)
        return accuracy

def main():
    config = Config()
    model = MainModel(config)
    train_loader, val_loader, test_loader = load_data(config)
    model.train(train_loader, val_loader)
    accuracy = model.evaluate(test_loader)
    logging.info(f'Model accuracy: {accuracy:.4f}')
    save_model(model, config)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()