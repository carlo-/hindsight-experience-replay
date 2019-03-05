from datetime import timedelta

import gym
from tqdm import tqdm

from ddpg_agent import DdpgHer
from train import OUT_DIR


def demo(env, agent, steps=10000, reset_on_success=True):
    obs = env.reset()
    for _ in range(steps):
        action = agent.predict(obs)
        obs, reward, done, _ = env.step(action)
        env.render()
        if reward == 0.0:
            print('Success!')
            done = done or reset_on_success
        if done:
            obs = env.reset()


def record_successes(env, agent, *, n=10, output_dir, single_file=True):
    import imageio
    frames = []
    all_frames = []
    obs = env.reset()

    while len(all_frames) < n:

        if obs['desired_goal'][2] < 0.48:
            obs = env.reset()
            continue

        action = agent.predict(obs)
        obs, reward, done, _ = env.step(action)
        img = env.render(mode='rgb_array', rgb_options=dict(camera_id=-1))
        frames.append(img)
        if reward == 0.0:
            print('Success!')
            all_frames.append(frames)
            done = True
        if done:
            frames = []
            obs = env.reset()

    if single_file:
        all_frames = [sum(all_frames, [])]
    for frames in all_frames:
        if single_file:
            file_path = f'{output_dir}/episodes.gif'
        else:
            file_path = f'{output_dir}/episode{len(all_frames)+1}.gif'
        imageio.mimwrite(file_path, frames, fps=30)


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
    # local_dir = f'{OUT_DIR}/fetch_test/mordor'

    env = gym.make(
        'HandPickAndPlace-v0',
        ignore_rotation_ctrl=True,
        ignore_target_rotation=True,
        success_on_grasp_only=False,
        randomize_initial_arm_pos=True,
        randomize_initial_object_pos=True,
        distance_threshold=0.05
    )
    local_dir = f'{OUT_DIR}/hand_pp_parallel_arm_grasp_init/mordor'

    # plot_progress(f'{local_dir}/progress.csv')
    agent = DdpgHer.load(env, local_dir, epoch=60)
    demo(env, agent, reset_on_success=True)
    # evaluate(env, agent, episodes=1000)
    # record_successes(env, agent, output_dir=local_dir, n=10, single_file=True)


def generate_hand_pp_demonstrations():
    from gym.agents.shadow_hand import HandPickAndPlaceAgent
    from utils import demonstrations_from_agent
    env = gym.make(
        'HandPickAndPlace-v0',
        ignore_rotation_ctrl=True,
        ignore_target_rotation=True,
        randomize_initial_arm_pos=True,
        randomize_initial_object_pos=True,
        distance_threshold=0.05
    )
    agent = HandPickAndPlaceAgent(env)

    def should_skip_episode(obs):
        if obs['desired_goal'][2] < 0.48:
            # desired goal is too close to the table
            return True
        return False

    demonstrations_from_agent(env, agent, n=100, output_path='./out/demonstrations/hand_demo_100.pkl',
                              render=True, skip_episode=should_skip_episode)


if __name__ == '__main__':
    main()
