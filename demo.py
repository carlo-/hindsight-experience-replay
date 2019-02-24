import gym
from tqdm import tqdm

from ddpg_agent import DdpgHer
from train import OUT_DIR


def demo(env, agent, steps=10000):
    obs = env.reset()
    for _ in range(steps):
        action = agent.predict(obs)
        obs, reward, done, _ = env.step(action)
        env.render()
        if done:
            obs = env.reset()


def evaluate(env, agent, episodes=200):
    success_rate = 0.0
    for _ in tqdm(range(episodes)):
        obs = env.reset()
        while True:
            action = agent.predict(obs)
            obs, reward, done, info = env.step(action)
            success = info['is_success']
            if done or success == 1.0:
                success_rate += success
                break
    success_rate /= episodes
    print('success_rate', success_rate)


def main():

    # env = gym.make('FetchPickAndPlace-v1')
    # local_dir = f'{OUT_DIR}/test3'

    env = gym.make('HandPickAndPlace-v0', ignore_rotation_ctrl=True, ignore_target_rotation=True)
    local_dir = f'{OUT_DIR}/hand_pick_and_place_obs_ls_no_rot'

    agent = DdpgHer.load(env, local_dir, epoch=44)

    demo(env, agent)


if __name__ == '__main__':
    main()
