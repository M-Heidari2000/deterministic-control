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
    

class PosteriorModel(nn.Module):
    """
        s_t = g(o1:t, a1:t-1)
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        state_dim: int,
        rnn_hidden_dim: int,
        rnn_input_dim: int,
    ):
        super().__init__()

        # RNN hidden at time t summarizes o1:t-1, a1:t-2
        self.rnn = nn.GRUCell(
            input_size=rnn_input_dim,
            hidden_size=rnn_hidden_dim,
        )

        self.fc_obs_action = nn.Sequential(
            nn.Linear(observation_dim + action_dim, rnn_input_dim),
            nn.ReLU(),
        )

        self.state_head = nn.Linear(rnn_hidden_dim, state_dim)

        self.rnn_hidden_dim = rnn_hidden_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rnn_input_dim = rnn_input_dim

    def forward(
        self,
        prev_rnn_hidden,
        prev_action,
        observation,
    ):
        rnn_input = self.fc_obs_action(
            torch.cat([observation, prev_action], dim=1)
        )
        rnn_hidden = self.rnn(rnn_input, prev_rnn_hidden)

        state = self.state_head(rnn_hidden)

        return state