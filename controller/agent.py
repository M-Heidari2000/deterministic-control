import torch
import einops
from torchrl.modules import TruncatedNormal


class CEMAgent:
    """
        action planning by Cross Entropy Method (CEM)
    """

    def __init__(
        self,
        env,
        transition_model,
        encoder_model,
        reward_model,
        planning_horizon: int,
        num_iterations: int,
        num_candidates: int,
        num_elites: int,
    ):
        self.transition_model = transition_model
        self.encoder_model = encoder_model
        self.reward_model = reward_model
        self.num_iterations = num_iterations
        self.num_candidates = num_candidates
        self.num_elites = num_elites
        self.planning_horizon = planning_horizon

        self.device = next(encoder_model.parameters()).device
        self.action_low = torch.as_tensor(env.action_space.low, device=self.device).unsqueeze(0)
        self.action_high = torch.as_tensor(env.action_space.high, device=self.device).unsqueeze(0) 

    def __call__(self, obs):

        # convert o_t to a torch tensor and add a batch dimension
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)

        # no learning takes place here
        with torch.no_grad():
            initial_state = self.encoder_model(obs)

            # initialize action distribution ~ N(0, I)
            action_dist = TruncatedNormal(
                0.5 * (self.action_high + self.action_low) + torch.zeros(
                    (self.planning_horizon, self.action_low.shape[1]),
                    device=self.device
                ),
                0.2 * (self.action_high - self.action_low) * torch.ones(
                    (self.planning_horizon, self.action_low.shape[1]),
                    device=self.device
                ),
                low=self.action_low,
                high=self.action_high
            )

            # iteratively improve action distribution with CEM
            for _ in range(self.num_iterations):
                # sample action candidates
                # reshape to (planning_horizon, num_candidates, action_dim) for parallel exploration
                action_candidates = action_dist.sample([self.num_candidates])
                action_candidates = einops.rearrange(action_candidates, "n h a -> h n a")

                state = initial_state.repeat([self.num_candidates, 1])
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
                action_dist.scale = std + 1e-6
            
            # return only mean of the first action (MPC)
            action = mean[0]

        return action.cpu().numpy()