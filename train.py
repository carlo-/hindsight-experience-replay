import sys
import os

from mpi4py import MPI

from ddpg_agent import DdpgHer
from utils import TensorboardReporter


OUT_DIR = os.path.join(os.path.dirname(__file__), 'out/her_torch')
os.makedirs(OUT_DIR, exist_ok=True)


def init_env(env_id, env_config=None):
    import gym
    env_config = env_config or dict()
    return gym.make(env_id, **env_config)


def train(config: dict=None, reporter=None):
    env = init_env(config['env'], config.get('env_config'))
    print(f'Environment with rank {MPI.COMM_WORLD.Get_rank()} ready.')
    agent = DdpgHer(env, config, reporter)
    agent.train()


def train_mpi(config: dict=None, reporter=None):

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
        train(config, reporter)

    print('Waiting...')
    comm.barrier()
    if is_master:
        print('Done.')


def simple_reporter(**kwargs):
    print(kwargs)


def main(spawn_children=False):

    local_dir = f'{OUT_DIR}/hand_reach_no_rot'
    reporter = TensorboardReporter(log_dir=local_dir)
    config = dict(
        env="MovingHandReach-v0",
        env_config=dict(ignore_rotation_ctrl=True),
        n_workers=6,
        n_epochs=100,
        checkpoint_freq=10,
        local_dir=local_dir
    )

    if spawn_children:
        train_mpi(config, reporter)
    else:
        train(config, reporter)


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    main(spawn_children=False)
