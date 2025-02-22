import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, obs_shape, act_shape):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, act_shape)
        )

    def forward(self, x):
        out = self.fc(x)
        return out
