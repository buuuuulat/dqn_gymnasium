import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import collections

import dqn_model

from torch.utils.tensorboard import SummaryWriter


DEVICE = torch.device('mps' if torch.mps.is_available() else 'cpu')
ENV_NAME = 'Acrobot-v1'

GAMMA = 0.99
BATCH_SIZE = 64
REPLAY_SIZE = 5000
REPLAY_START_SIZE = 5000
LEARNING_RATE = 3e-4
SYNC_TARGET_FRAMES = 2000
REWARD_BOUND = 0

EPSILON_DECAY_LAST_FRAME = 50000
EPSILON_START = 1
EPSILON_END = 0.01


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
        states, actions, rewards, terminated, truncated, new_states = zip(*[self.buffer[idx] for idx in indices])
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(terminated),
            np.array(truncated),
            np.array(new_states)
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
    def play_step(self, net, epsilon=0.0, device=DEVICE):
        done_reward = None

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            states_t = torch.as_tensor(np.array([self.state]), dtype=torch.float32).to(device)
            q_vals_t = net(states_t)
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

    def calc_loss(self, batch, net, tgt_net, device=DEVICE):
        states, actions, rewards, terminated, truncated, new_states = batch

        states_t = torch.as_tensor(states).to(device)
        next_states_t = torch.as_tensor(new_states).to(device)
        actions_t = torch.as_tensor(actions).to(device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32).to(device)
        done_finished = torch.BoolTensor(terminated).to(device)

        state_action_values = net(states_t).gather(1, actions_t.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = tgt_net(next_states_t).max(1)[0]
            next_state_values[done_finished] = 0.0

        expected_state_action_values = rewards_t + GAMMA * next_state_values
        return nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__ == '__main__':
    env = gym.make(ENV_NAME)

    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device=DEVICE)
    net.train()

    tgt_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device=DEVICE)
    tgt_net.train()

    writer = SummaryWriter()

    buffer = ExperienceBuffer(capacity=REPLAY_SIZE)
    agent = Agent(env=env, exp_buffer=buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    best_m_reward = None

    frame_idx = 0

    while True:
        frame_idx += 1
        epsilon = max(EPSILON_END, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

        reward = agent.play_step(net, epsilon, device=DEVICE)
        if reward is not None:
            total_rewards.append(reward)
            m_reward = np.mean(total_rewards[-100:])
            print(f'{frame_idx}: done {len(total_rewards)} games, reward {m_reward}, eps {epsilon}')

            writer.add_scalar('epsilon', epsilon, frame_idx)
            writer.add_scalar('reward_100', m_reward, frame_idx)
            writer.add_scalar('reward', reward, frame_idx)

            if best_m_reward is None or best_m_reward < m_reward:
                torch.save(net.state_dict(), './model_saves/' + ENV_NAME + 'best_%.0f.dat' % m_reward)
                if best_m_reward is not None:
                    print(f'Best reward updated! %.3f -> %.3f' % (best_m_reward, m_reward))
                best_m_reward = m_reward
            if best_m_reward > REWARD_BOUND:
                print(f'Solved in {frame_idx} iterations!')
                break

        if len(buffer) < REPLAY_START_SIZE:
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample()
        loss_t = agent.calc_loss(batch, net, tgt_net, device=DEVICE)
        loss_t.backward()
        optimizer.step()

        if frame_idx % 1000 == 0:
            torch.mps.empty_cache()
