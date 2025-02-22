import collections
import torch
import gymnasium as gym
import numpy as np

import dqn_model

ENV_NAME = 'Acrobot-v1'

if __name__ == '__main__':
    while True:
        env = gym.make(ENV_NAME, render_mode='human')

        net = dqn_model.DQN(env.observation_space.shape, env.action_space.n)
        net.load_state_dict(torch.load('./model_saves/Acrobot-v1best_-79.dat'))
        net.eval()

        state, info = env.reset()
        total_reward = 0.0
        c = collections.Counter()

        while True:
            state_t = torch.FloatTensor(state).unsqueeze(0)

            with torch.no_grad():
                q_vals = net(state_t)

            action = torch.argmax(q_vals).item()
            c[action] += 1

            new_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            state = new_state

            if terminated or truncated:
                break

        print(f'Total reward: {total_reward}')
        print(f'Action counts: {c}')
        env.close()