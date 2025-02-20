from functools import total_ordering

import gymnasium as gym
import torch
import numpy as np
import collections

from fontTools.misc.plistlib import totree

DEViCE = torch.device('mps' if torch.mps.is_available() else 'cpu')

Experience = collections.namedtuple('Experience',
                                    field_names=['state', 'action', 'reward', 'terminated', 'truncated', 'new_state'])


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size=32):
        indices =  np.random.choice(len(self.buffer), batch_size)
        states, actions, rewards, terminated, truncated, new_state = zip(*[self.buffer[idx] for idx in indices])
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(terminated),
            np.array(truncated),
            np.array(new_state)
        )


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state, info =  self.env.reset()
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device=DEViCE):
        done_reward = None

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            states_t = torch.as_tensor([self.state]).to(device=DEViCE)
            q_vals_t = net(states_t.shape)
            _, act_t = torch.max(q_vals_t, dim=1)
            action = int(act_t.item())

        new_state, reward, terminated, truncated, infp = self.env.step(action)
        self.total_reward += reward

        exp = Experience(
            self.state,
            action,
            reward,
            terminated,
            truncated,
            new_state
        )

        self.exp_buffer.append(exp)
        self.state = new_state
        if terminated or truncated:
            done_reward = self.total_reward
            self._reset()
        return done_reward

    def calc_loss(self, batch, net, tgt_net, device=DEViCE):
        pass
