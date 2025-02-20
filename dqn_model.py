import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, obs_shape, act_shape):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_shape, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, act_shape)
        )

    def forward(self, x):
        out = self.fc(x)
        return out
