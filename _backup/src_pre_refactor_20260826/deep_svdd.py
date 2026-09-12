import torch
import torch.nn as nn


class DeepSVDD(nn.Module):

    def __init__(self, input_dim=69, latent_dim=16):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)