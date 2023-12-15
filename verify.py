import tianshou as ts
from gymnasium.wrappers import FlattenObservation
import gymnasium as gym
from tianshou.utils.net.common import Net
import rl_envs
import torch
from tianshou.data import Batch

env = FlattenObservation(gym.make('rl_envs/FifteenPuzzle-v0',render_mode='human'))

state_shape = env.observation_space.shape
action_shape = env.action_space.n
net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=[128, 128, 128])

lr, epoch, batch_size = 1e-3, 10, 64
gamma, n_step, target_freq = 0.9, 3, 320
optim = torch.optim.Adam(net.parameters(), lr=lr)
policy = ts.policy.DQNPolicy(
    model=net,
    optim=optim,
    discount_factor=gamma,
    action_space=env.action_space,
    estimation_step=n_step,
    target_update_freq=target_freq
)

policy.load_state_dict(torch.load('dqn.pth'))

policy.eval()
policy.set_eps(0.05)
collector = ts.data.Collector(policy, env, exploration_noise=True)
print(collector.collect(n_episode=1, render=1 / 35))
