
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging

logging.basicConfig(level=logging.INFO)

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


def train_linear(model, loader, lr, weight_decay, l1_lambda, epochs, patience):
    """
    Обучение линейной регрессии с L1, L2 и early stopping
    """
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    best_loss = float('inf')
    counter = 0

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for X, y in loader:
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)

            if l1_lambda > 0:
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss = loss + l1_lambda * l1_norm
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        logging.info(f"Epoch {epoch}, Loss: {avg:.4f}")

        if avg < best_loss:
            best_loss = avg
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                logging.info("Early stopping")
                break

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

if __name__ == '__main__':
    pass