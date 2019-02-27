from datetime import timedelta

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


def plot_progress(csv_path):
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    df = pd.read_csv(csv_path)
    success_rate = df['test_success_rate'].values
    steps_per_epoch = df['env_steps'][0]

    fig, ax1 = plt.subplots()
    ax1.plot(df['training_iteration'].values, success_rate)
    ax1.set_ylabel('Success rate')
    ax1.set_xlabel('Epoch ({} env. steps each)'.format(steps_per_epoch))

    ax2 = ax1.twiny()
    ax2.plot(df['total_time'].values, success_rate).pop(0).remove()

    def format_date(x, pos=None):
        return str(timedelta(seconds=x))

    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
    for tick in ax2.get_xticklabels():
        tick.set_rotation(45)

    plt.show()


def main():

    # env = gym.make('FetchPickAndPlace-v1')
    # local_dir = f'{OUT_DIR}/test3'

    env = gym.make('HandPickAndPlace-v0', ignore_rotation_ctrl=True, ignore_target_rotation=True, success_on_grasp_only=True)
    local_dir = f'{OUT_DIR}/hand_pp_grasp_only/mordor'

    plot_progress(f'{local_dir}/progress.csv')
    agent = DdpgHer.load(env, local_dir, epoch=100)
    demo(env, agent)


if __name__ == '__main__':
    main()
