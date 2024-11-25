import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal


class ObservationEncoder(nn.Module):
    """
        Embeds observations into vectors
    """
    def __init__(self, obs_dim: int, obs_embedding_dim: int):
        super().__init__()
        self.mlp_layers = nn.Sequential(
            nn.Linear(obs_dim, 2*obs_dim),
            nn.ReLU(),
            nn.Linear(2*obs_dim, obs_embedding_dim),
        )

    def forward(self, obs):
        embedded_obs = self.mlp_layers(obs)
        return embedded_obs
    
