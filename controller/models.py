import torch
from torch import nn


class ObservationEncoder(nn.Module):
    """
        Embeds observations into vectors
    """
    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        obs_embedding_dim: int
    ):
        super().__init__()
        self.mlp_layers = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_embedding_dim),
        )

    def forward(self, obs):
        embedded_obs = self.mlp_layers(obs)
        return embedded_obs
    

class RewardModel(nn.Module):
    """
        p(r_t | h_t, a_t)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int
    ):
        super().__init__()
        self.mlp_layers = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        reward = self.mlp_layers(
            torch.cat([state, action], dim=1)
        )
        return reward
    

class ObservationModel(nn.Module):
    """
        p(o_t | h_t)
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        obs_dim: int,
    ):
        super().__init__()
        self.mlp_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )

    def forward(self, state):
        obs = self.mlp_layers(state)
        return obs
    

class TransitionModel(nn.Module):
    """
        h_{t+1} = f(h_t, a_t)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.rnn = nn.GRUCell(
            input_size=action_dim,
            hidden_size=state_dim,
        )


    def forward(self, state, action):
        next_state = self.rnn(state, action)
        return next_state