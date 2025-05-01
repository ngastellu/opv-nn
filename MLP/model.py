import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FingerprintMLPRegressor(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.input_dim = configs.input_dim
        self.dropout_prob = configs.dropout

        layers = []

        hidden_dims = [2 * self.input_dim, self.input_dim // 2]
        prev_dim = self.input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())  # or nn.GELU() for smoother activation
            layers.append(nn.Dropout(self.dropout_prob))
            prev_dim = hidden_dim

        # Final output layer: single unit for regression
        layers.append(nn.Linear(prev_dim, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze(-1)  # return shape [batch_size]


