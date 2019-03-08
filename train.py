import sys
import os
from datetime import datetime

from mpi4py import MPI

from ddpg_agent import DdpgHer
from utils import MultiReporter


OUT_DIR = os.path.join(os.path.dirname(__file__), 'out/her_torch')
os.makedirs(OUT_DIR, exist_ok=True)


def init_env(env_id, env_config=None):
    import gym
    env_config = env_config or dict()
    return gym.make(env_id, **env_config)


def train(config: dict):

    seed = config.get('seed', 42)
    rank = MPI.COMM_WORLD.Get_rank()
    rank_seed = seed + 1000000 * rank

    if rank == 0:
        local_dir = config['local_dir']
        now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        local_dir = f'{local_dir}/{now}'
        os.makedirs(local_dir, exist_ok=False)
        config['local_dir'] = local_dir
        reporter = MultiReporter(log_dir=local_dir)
    else:
        reporter = None

    env = init_env(config['env'], config.get('env_config'))
    print(f'Environment with rank {rank} ready.')
    agent = DdpgHer(env, config, reporter)
    agent.seed(rank_seed)
    print(f'Worker with rank {rank} and seed {rank_seed} ready.')
    agent.train()


def train_mpi(config: dict):

    comm = MPI.Comm.Get_parent()
    if comm == MPI.COMM_NULL:
        comm = MPI.COMM_WORLD
        is_master = True
    else:
        comm = comm.Merge()
        is_master = False

    if is_master:
        n_cpus = max(config.get('n_workers', 1), 1)
        comm = comm.Spawn(sys.executable, args=[__file__], maxprocs=n_cpus).Merge()

    config = comm.bcast(config, root=0)
    print(f'Worker with rank {MPI.COMM_WORLD.Get_rank()} {" (master)" if is_master else ""} ready.')

    if not is_master:
        train(config)

    print('Waiting...')
    comm.barrier()
    if is_master:
        print('Done.')


def simple_reporter(**kwargs):
    print(kwargs)


def main(spawn_children=False):

    local_dir = f'{OUT_DIR}/hand_pp_lfd'
    config = dict(
        env="HandPickAndPlace-v0",
        env_config=dict(
            ignore_rotation_ctrl=True,
            ignore_target_rotation=True,
            success_on_grasp_only=False,
            randomize_initial_arm_pos=True,
            randomize_initial_object_pos=True,
            distance_threshold=0.05,
            grasp_state=True,
            grasp_state_reset_p=0.2,
            target_in_the_air_p=1.0
        ),
        n_epochs=500,
        checkpoint_freq=1,
        local_dir=local_dir,
        q_filter=True,
        demo_batch_size=128,
        demo_file='./demonstrations/hand_demo_500_sim.pkl',
        num_demo=500,
    )

    if spawn_children:
        train_mpi(config)
    else:
        train(config)


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    main(spawn_children=False)
