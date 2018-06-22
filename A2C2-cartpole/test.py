'''
Intuition: 2 Critic Network for 2 Objectives (reward signals)
Environment: Cartpole from gym with an additional objective : move to position x=1.
Algorithm: A2C(2) with PPO
'''

import numpy as np
import gym
from A2C2 import A2C2_d
from A2C2 import Settings

import gym.envs.registration as reg
reg.register(
    id='CartPole-v2',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=300,
    reward_threshold=270.0,
)
env = gym.make('CartPole-v2')

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
settings = Settings(state_dim, action_dim, writer_f=False)

agent = A2C2_d(settings)

# Choose 1-Critic or 2-Critic Arch.
def reward_comp_2critic(r1, r2): return [r1, r2]
def reward_comp_1critic(r1, r2): return [r1+r2, 0]
reward_comp = reward_comp_2critic

# Training process
EPS = 4000
for ieps in range(EPS):
    done = False
    state = env.reset()
    agent.buffer_clear()

    rs = []

    while not done:
        if ieps % 100 == 0:
            env.render()
        action, value = agent.forward(state)

        next_state, reward, done, _ = env.step(action)

        reward_org = reward - 0.99      # roughly unbias
        # reward_pos = (1.5 - (1. - state[0])**2) * 0.5
        reward_pos = 0.1 if np.abs(1. - state[0]) < 0.2 else 0.     # additional objective

        reward = reward_comp(reward_org, reward_pos)
        rs.append(reward)

        agent.buffer_append(state, action, reward, value, next_state)
        if len(agent.rewards) >= 100 or done:
            closs = agent.train(done)
        if done:
            rs = np.array(rs)
            print('eps:{} | r:{:.2f}, {:.2f} | closs:{:.2f}'.format(ieps, np.sum(rs, axis=0)[0], np.average(rs, axis=0)[1], closs))

        state = next_state


