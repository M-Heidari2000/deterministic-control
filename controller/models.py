import torch
import torch.nn as nn
from typing import Optional
from torch.distributions import Normal


class ObservationModel(nn.Module):
    """
        p(o_t | s_t)
        constant variance
    """
    def __init__(
        self,
        state_dim: int,
        observation_dim: int,
        hidden_dim: Optional[int]=None,
    ):
        super().__init__()

        hidden_dim = hidden_dim if hidden_dim is not None else state_dim * 2
        
        self.mlp_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, observation_dim),
        )

    def forward(self, state):
        return self.mlp_layers(state)
    

class RewardModel(nn.Module):
    """
        p(r_t | s_t)
        constant variance
    """
    def __init__(
        self,
        state_dim: int,
        hidden_dim: Optional[int]=None,
    ):
        super().__init__()

        hidden_dim = hidden_dim if hidden_dim is not None else 2*state_dim

        self.mlp_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.mlp_layers(state)
    

class TransitionModel(nn.Module):
    """
        s_t = f(s_{t-1}, a_{t-1})
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: Optional[int]=None,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.state_dim = state_dim

        hidden_dim = (
            hidden_dim if hidden_dim is not None else 2*(state_dim + action_dim)
        )

        self.mlp_layers = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, prev_state, prev_action):
        state = self.mlp_layers(
            torch.cat([prev_state, prev_action], dim=1)
        )
        return state
    

class EncoderModel(nn.Module):
    """
        s_t = g(ot)
    """

    def __init__(
        self,
        observation_dim: int,
        state_dim: int,
        hidden_dim: Optional[int]=None,
    ):
        super().__init__()

        hidden_dim = hidden_dim if hidden_dim is not None else 2*observation_dim

        self.mlp_layers = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(
        self,
        observation,
    ):
        return self.mlp_layers(observation)