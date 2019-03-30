from datetime import timedelta

import gym
from tqdm import tqdm

from ddpg_agent import DdpgHer
from train import OUT_DIR

REMOTE_OUT_DIR = f'/run/user/1000/gvfs/sftp:host=mordor.csc.kth.se,port=2222,user=carlora/home/carlora/thesis/exp/hindsight-experience-replay/out/her_torch/'


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
        randomize_initial_arm_pos=True,
        randomize_initial_object_pos=True,
        distance_threshold=0.05,
        object_id='small_sphere',
    )
    local_dir = f'{REMOTE_OUT_DIR}/hand_pp_small_sphere_lfd'

    # env = gym.make('YumiBar-v1', reward_type='sparse')
    # local_dir = f'{REMOTE_OUT_DIR}/yumi_bar_test4'

    import glob
    local_dir = glob.glob(local_dir + '/*/checkpoints')[0]

    # plot_progress(f'{local_dir}/progress.csv')
    agent = DdpgHer.load(env, local_dir)
    demo(env, agent, reset_on_success=False)
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
        distance_threshold=0.05,
        object_id='teapot',
    )
    agent = HandPickAndPlaceAgent(env)

    def should_skip_episode(obs):
        if obs['desired_goal'][2] < 0.48:
            # desired goal is too close to the table
            return True
        return False

    demonstrations_from_agent(env, agent, n=500, output_path='./demonstrations/hand_demo_500_teapot.pkl',
                              render=False, skip_episode=should_skip_episode)


def generate_yumi_reach_demonstrations():
    from gym.agents.yumi import YumiReachAgent
    from utils import demonstrations_from_agent
    env = gym.make('YumiReachTwoArms-v0', reward_type='sparse')
    agent = YumiReachAgent(env)
    file_path = './demonstrations/yumi_reach_two_arms_100.pkl'
    demonstrations_from_agent(env, agent, n=100, output_path=file_path, render=False)


def generate_yumi_bar_demonstrations():
    from gym.agents.yumi import YumiBarAgent
    from utils import demonstrations_from_agent
    env = gym.make('YumiBar-v1', reward_type='sparse', randomize_initial_object_pos=True)
    agent = YumiBarAgent(env)
    file_path = './demonstrations/yumi_bar_300.pkl'
    demonstrations_from_agent(env, agent, n=300, output_path=file_path, render=False)


if __name__ == '__main__':
    main()
