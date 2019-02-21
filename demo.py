import gym

from ddpg_agent import DdpgHer
from train import OUT_DIR


def main():

    # env = gym.make('FetchPickAndPlace-v1')
    # local_dir = f'{OUT_DIR}/test3'

    env = gym.make('MovingHandReach-v0', ignore_rotation_ctrl=True, ignore_target_rotation=True)
    local_dir = f'{OUT_DIR}/hand_reach_no_rot'

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
