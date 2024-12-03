import torch
import einops
from torch.distributions import Normal


class CEMAgent:
    """
        action planning by Cross Entropy Method (CEM)
    """

    def __init__(
        self,
        transition_model,
        encoder_model,
        reward_model,
        action_low,
        action_high,
        planning_horizon: int,
        num_iterations: int,
        num_candidates: int,
        num_elites: int,
    ):
        self.transition_model = transition_model
        self.encoder_model = encoder_model
        self.reward_model = reward_model
        self.action_low = action_low
        self.action_high = action_high
        self.num_iterations = num_iterations
        self.num_candidates = num_candidates
        self.num_elites = num_elites
        self.planning_horizon = planning_horizon

        self.device = next(encoder_model.parameters()).device

    def __call__(self, obs):

        # convert o_t and a_{t-1} to a torch tensor and add a batch dimension
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)


        # no learning takes place here
        with torch.no_grad():
            state = self.encoder_model(obs)

            # initialize action distribution ~ N(0, I)
            action_dist = Normal(
                0.5 * (self.action_high + self.action_low) + torch.zeros(
                    (self.planning_horizon, self.posterior_model.action_dim),
                    device=self.device
                ),
                0.2 * (self.action_high - self.action_low) * torch.ones(
                    (self.planning_horizon, self.posterior_model.action_dim),
                    device=self.device
                ),
            )

            # iteratively improve action distribution with CEM
            for _ in range(self.num_iterations):
                # sample action candidates
                # reshape to (planning_horizon, num_candidates, action_dim) for parallel exploration
                action_candidates = action_dist.sample([self.num_candidates]).clamp(self.action_low, self.action_high)
                action_candidates = einops.rearrange(action_candidates, "n h a -> h n a")

                state = state.repeat([self.num_candidates, 1])
                total_predicted_reward = torch.zeros(self.num_candidates, device=self.device)

                # start generating trajectories starting from s_t using transition model
                for t in range(self.planning_horizon):
                    total_predicted_reward += self.reward_model(state=state).squeeze()
                    # get next state from our prior (transition model)
                    state = self.transition_model(
                        prev_state=state,
                        prev_action=action_candidates[t],
                    )

                # find elites
                elite_indexes = total_predicted_reward.argsort(descending=True)[:self.num_elites]
                elites = action_candidates[:, elite_indexes, :]

                # fit a new distribution to the elites
                mean = elites.mean(dim=1)
                std = elites.std(dim=1, unbiased=False)
                action_dist.loc = mean
                action_dist.scale = std
            
            # return only mean of the first action (MPC)
            action = mean[0]

        return action.cpu().numpy()
    
    def reset(self):
        self.rnn_hidden = torch.zeros(1, self.posterior_model.rnn_hidden_dim, device=self.device)