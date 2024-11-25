import torch
import einops
from torch.distributions import Normal
from .models import (
    ObservationEncoder,
    RewardModel,
    TransitionModel
)


class CEMAgent:
    """
        Action planning by Cross Entropy Method (CEM)
    """
    def __init__(
        self,
        obs_encoder: ObservationEncoder,
        reward_model: RewardModel,
        transition_model: TransitionModel,
        horizon: int,
        N_iterations: int,
        N_candidates: int,
        N_elites: int
    ):
        self.obs_encoder = obs_encoder,
        self.reward_model = reward_model,
        self.transition_model = transition_model

        self.horizon = horizon
        self.N_iterations = N_iterations
        self.N_candidates = N_candidates
        self.N_elites = N_elites

        self.device = next(self.reward_model.parameters()).device
        # Initialize the first state to a zero vector
        self.state = torch.zeros(1, self.transition_model.state_dim, device=self.device)

    def __call__(self, obs):
        with torch.no_grad():
            obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
            embedded_obs = self.obs_encoder(obs)

            # Initialize action distribution ~ N(0, I)
            action_dist = Normal(
                torch.zeros(self.horizon, self.transition_model.action_dim, device=self.device),
                torch.ones(self.horizon, self.transition_model.action_dim, device=self.device),
            )

            for _ in range(self.N_iterations):
                # Sample action sequences
                action_candidates = action_dist.sample([self.N_candidates])
                action_candidates = einops.rearrange(action_candidates, "n h a -> h n a")

                total_predicted_reward = torch.zeros(self.N_candidates, device=self.device)
                state = self.state.repeat([self.N_candidates, 1])

                # Compute total predicted reward by running the actions
                for t in range(self.horizon):
                    total_predicted_reward += self.reward_model(
                        state=state,
                        action=action_candidates[t],
                    ).squeeze()
                    next_state = self.transition_model(
                        state=state,
                        action=action_candidates[t],
                    )
                    state = next_state

                # Find elites
                elite_indexes = total_predicted_reward.argsort(descending=True)[:self.N_elites]
                elites = action_candidates[:, elite_indexes, :]

                # Fit a new distribution to the elites
                action_dist.loc = elites.mean(dim=1)
                action_dist.scale = elites.std(unbiased=False)

        
            action = action_dist.loc[0]
            return action.cpu().numpy()
    
    def reset(self):
        self.state = torch.zeros(1, self.transition_model.state_dim, device=self.device)