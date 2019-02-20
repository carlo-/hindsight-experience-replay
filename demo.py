import torch
import gym

from ddpg_agent import DdpgHer
from train import OUT_DIR


def main():

    env = gym.make('FetchPickAndPlace-v1')
    local_dir = f'{OUT_DIR}/test1'
    agent = DdpgHer.load(env, local_dir)

    obs = env.reset()
    for _ in range(10000):
        action = agent.predict(obs)
        obs, reward, done, _ = env.step(action)
        env.render()
        if done:
            obs = env.reset()


if __name__ == '__main__':
    main()
