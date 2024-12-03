import os
import time
import json
import torch
import einops
import numpy as np
import gymnasium as gym
from pathlib import Path
from datetime import datetime
from torch.distributions.kl import kl_divergence
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard.writer import SummaryWriter
from .agent import CEMAgent
from .memory import ReplayBuffer
from .wrappers import RepeatActionWrapper
from .configs import TrainConfig
from .models import (
    ObservationModel,
    RewardModel,
    TransitionModel,
    EncoderModel,
)


def train(env: gym.Env, config: TrainConfig):

    # prepare logging
    log_dir = Path(config.log_dir) / datetime.now().strftime("%Y%m%d_%H%M")
    os.makedirs(log_dir, exist_ok=True)
    with open(log_dir / "args.json", "w") as f:
        json.dump(config.dict(), f)
    
    writer = SummaryWriter(log_dir=log_dir)

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # wrap env
    env = RepeatActionWrapper(env=env, skip=config.action_repeat)

    # define replay buffer
    replay_buffer = ReplayBuffer(
        capacity=config.buffer_capacity,
        observation_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
    )

    # define models and optimizer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    observation_model = ObservationModel(
        state_dim=config.state_dim,
        observation_dim=env.observation_space.shape[0],
    ).to(device)

    reward_model = RewardModel(
        state_dim=config.state_dim,
    ).to(device)

    transition_model = TransitionModel(
        state_dim=config.state_dim,
        action_dim=env.action_space.shape[0],
    ).to(device)

    encoder_model = EncoderModel(
        observation_dim=env.observation_space.shape[0],
        state_dim=config.state_dim,
    ).to(device)

    all_params = (
        list(observation_model.parameters()) +
        list(reward_model.parameters()) +
        list(transition_model.parameters()) +
        list(encoder_model.parameters())
    )  

    cem_agent = CEMAgent(
        env,
        transition_model=transition_model,
        encoder_model=encoder_model,
        reward_model=reward_model,
        planning_horizon=config.planning_horizon,
        num_iterations=config.num_iterations,
        num_candidates=config.num_candidates,
        num_elites=config.num_elites,
    )

    optimizer = torch.optim.Adam(all_params, lr=config.lr, eps=config.eps)

    # collect initial experience with random actions
    for episode in range(config.seed_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.push(obs, action, reward, done)
            obs = next_obs

    # main training loop
    for episode in range(config.seed_episodes, config.all_episodes):

        # collect experience
        start = time.time()
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = cem_agent(obs=obs)
            action += np.random.normal(
                0,
                np.sqrt(config.action_noise_var),
                env.action_space.shape[0],
            )
            next_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            replay_buffer.push(obs, action, reward, done)
            obs = next_obs

        writer.add_scalar('total reward at train', total_reward, episode)
        print('episode [%4d/%4d] is collected. Total reward is %f' %
              (episode+1, config.all_episodes, total_reward))
        print('elasped time for interaction: %.2fs' % (time.time() - start))

        # update model parameters
        start = time.time()
        for update_step in range(config.collect_interval):
            observations, actions, rewards, _ = replay_buffer.sample(
                batch_size=config.batch_size,
                chunk_length=config.chunk_length,
            )

            observations = torch.as_tensor(observations, device=device)
            observations = einops.rearrange(observations, 'b l o -> l b o')
            actions = torch.as_tensor(actions, device=device)
            actions = einops.rearrange(actions, 'b l a -> l b a')
            rewards = torch.as_tensor(rewards, device=device)
            rewards = einops.rearrange(rewards, 'b l r -> l b r')
            
            # prepare Tensor to maintain states sequence and rnn hidden sequence
            states = torch.zeros(
                (config.chunk_length, config.batch_size, config.state_dim),
                device=device
            )

            state = encoder_model(observations[0])
            states[0] = state

            for t in range(0, config.chunk_length-1):
                state = transition_model(state, actions[t])

            flatten_states = states.reshape(-1, config.state_dim)
            
            recon_observations = observation_model(
                flatten_states,
            ).reshape(config.chunk_length, config.batch_size, env.observation_space.shape[0])

            recon_rewards = reward_model(
                flatten_states,
            ).reshape(config.chunk_length, config.batch_size, 1)

            obs_loss = 0.5 * mse_loss(
                recon_observations[1:],
                observations[1:],
                reduction='none'
            ).mean([0, 1]).sum()

            reward_loss = 0.5 * mse_loss(
                recon_rewards[1:],
                rewards[1:],
                reduction='none'
            ).mean([0, 1]).sum()

            loss = obs_loss + reward_loss
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(all_params, config.clip_grad_norm)
            optimizer.step()

            # print losses and add tensorboard
            print('update_step: %3d loss: %.5f, obs_loss: %.5f, reward_loss: %.5f'
                  % (update_step+1,
                     loss.item(), obs_loss.item(), reward_loss.item()))
            
            total_update_step = episode * config.collect_interval + update_step
            writer.add_scalar('overall loss', loss.item(), total_update_step)
            writer.add_scalar('obs loss', obs_loss.item(), total_update_step)
            writer.add_scalar('reward loss', reward_loss.item(), total_update_step)
        
        print('elasped time for update: %.2fs' % (time.time() - start))

        # test without exploration noise
        if (episode + 1) % config.test_interval == 0:
            obs, _ = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = cem_agent(obs=obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                obs = next_obs
                done = terminated or truncated

            writer.add_scalar('total reward at test', total_reward, episode)
            print('total test reward at episode [%4d/%4d] is %f' %
                (episode+1, config.all_episodes, total_reward))
            print('elasped time for test: %.2fs' % (time.time() - start))

     # save learned model parameters
    torch.save(observation_model.state_dict(), log_dir / "obs_model.pth")
    torch.save(reward_model.state_dict(), log_dir / "reward_model.pth")
    torch.save(transition_model.state_dict(), log_dir / "transition_model.pth")
    torch.save(encoder_model.state_dict(), log_dir / "encoder_model.pth")
    writer.close()
    
    return {"model_dir": log_dir}   