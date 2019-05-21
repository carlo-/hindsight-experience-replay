from datetime import timedelta

import gym
from tqdm import tqdm
import numpy as np

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
            print(success)
            if success == 1.0:
                success_rate += 1.0
                break
            if done:
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

    # env = gym.make(
    #     'HandPickAndPlace-v0',
    #     ignore_rotation_ctrl=True,
    #     ignore_target_rotation=True,
    #     randomize_initial_arm_pos=True,
    #     randomize_initial_object_pos=True,
    #     distance_threshold=0.05,
    #     object_id='original',
    # )
    # local_dir = f'{REMOTE_OUT_DIR}/hand_pp_lfd'

    # env = gym.make('YumiBar-v1', reward_type='sparse')
    # local_dir = f'{REMOTE_OUT_DIR}/yumi_bar_test4'

    # env = gym.make('YumiConstrained-v2', reward_type='sparse', render_poses=False)
    # local_dir = f'{OUT_DIR}/yumi_constrained_v2'

    # env = gym.make('YumiConstrained-v1', reward_type='sparse', render_poses=False)
    # local_dir = f'{REMOTE_OUT_DIR}/yumi_constrained_lfd'

    env = gym.make('YumiConstrained-v2', reward_type='sparse', render_poses=False, object_on_table=True)
    # local_dir = f'{OUT_DIR}/yumi_constr_push'
    local_dir = f'{OUT_DIR}/yumi_im_from_fetch_push'

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
    env = gym.make('YumiReachTwoArms-v1', reward_type='sparse')
    agent = YumiReachAgent(env)
    file_path = './demonstrations/yumi_reach_two_arms_700.pkl'
    demonstrations_from_agent(env, agent, n=100, output_path=file_path, render=False)


def generate_yumi_bar_demonstrations():
    from gym.agents.yumi import YumiBarAgent
    from utils import demonstrations_from_agent
    env = gym.make('YumiBar-v1', reward_type='sparse', randomize_initial_object_pos=True)
    agent = YumiBarAgent(env)
    file_path = './demonstrations/yumi_bar_300.pkl'
    demonstrations_from_agent(env, agent, n=300, output_path=file_path, render=False)


def generate_yumi_lift_demonstrations():
    from gym.agents.yumi import YumiLiftAgent
    from utils import demonstrations_from_agent

    def eval_success(obs_, reward_, done_, info_):
        return reward_ > -0.25

    env = gym.make('YumiLift-v1', randomize_initial_object_pos=False)
    agent = YumiLiftAgent(env)
    file_path = './demonstrations/yumi_lift_7000_fixed.pkl' # must be run with 7 workers to get 350 episodes
    demonstrations_from_agent(env, agent, n=1000, output_path=file_path, eval_success=eval_success,
                              render=False, store_sim_states=False)


def generate_yumi_constrained_demonstrations():
    from gym.agents.yumi import YumiConstrainedAgent
    from utils import demonstrations_from_agent
    env = gym.make('YumiConstrained-v1', reward_type='sparse')
    agent = YumiConstrainedAgent(env)
    file_path = './demonstrations/yumi_constrained_100.pkl'
    demonstrations_from_agent(env, agent, n=100, output_path=file_path, render=False, store_sim_states=False)


def generate_yumi_imitator_push_demonstrations():
    from utils import demonstrations_from_agent, import_thesis_package
    ms_thesis_path = import_thesis_package(remote=False)
    from playground.twin_vae import TwinVAE, TwinDataset, SimpleAutoencoder

    model = TwinVAE.load(f'{ms_thesis_path}/out/twin_yumi_v2_fetch_ae_resets_new_l/checkpoints/model_c1.pt',
                         net_class=SimpleAutoencoder)
    dataset = TwinDataset.load(f'{ms_thesis_path}/out/pp_yumi_v2_fetch_twin_dataset_10k.pkl')
    dataset.normalize()

    teacher_env = gym.make(
        'FetchPickAndPlaceLong-v1',
        reward_type='sparse',
        has_rotating_platform=True,
    )

    env = gym.make(
        'YumiConstrainedLong-v2',
        reward_type='sparse',
        render_poses=False,
        object_on_table=True,
        has_rotating_platform=True,
    )

    from gym.agents.yumi import YumiImitatorAgent
    from gym.agents.fetch import FetchPushAgent, FetchPickAndPlaceAgent

    teacher = FetchPickAndPlaceAgent(teacher_env)
    # teacher = FetchPushAgent(teacher_env)
    agent = YumiImitatorAgent(env, teacher_env=teacher_env, teacher_agent=teacher, a_scaler=dataset.a_scaler,
                              b_scaler=dataset.b_scaler, model=model)

    def should_skip_episode(obs):
        if np.linalg.norm(obs['desired_goal'] - obs['achieved_goal']) < 0.10:
            # object is already too close to the goal
            return True
        return False

    def step_callback():
        teacher_env.render()
        env.render()

    file_path = './demonstrations/yumi_imitator_from_fetch_push_platform_100.pkl'
    demonstrations_from_agent(env, agent, n=100, output_path=file_path, render=False, step_callback=step_callback,
                              store_sim_states=False, min_ep_length=15, skip_episode=should_skip_episode)


if __name__ == '__main__':
    generate_yumi_imitator_push_demonstrations()
