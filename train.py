import sys
import os

from mpi4py import MPI
import ray
import ray.tune as tune

from ddpg_agent import DdpgHer


OUT_DIR = os.path.join(os.path.dirname(__file__), 'out/ray')
os.makedirs(OUT_DIR, exist_ok=True)


def init_env(env_id, env_config=None):
    import gym
    env_config = env_config or dict()
    return gym.make(env_id, **env_config)


def train_ddpg_her(config: dict=None, reporter=None):

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
        env = init_env(config['env'], config.get('env_config'))
        print(f'Environment with rank {MPI.COMM_WORLD.Get_rank()} ready.')
        agent = DdpgHer(env, config, reporter)
        agent.train()

    print('Waiting...')
    comm.barrier()
    if is_master:
        print('Done.')


def train_with_ray():

    # This seems to be very slow and has problems with checkpoints...
    # Much better to use MPI by itself

    if MPI.Comm.Get_parent() == MPI.COMM_NULL:
        ray.init()
        tune.run_experiments(
            tune.Experiment(
                name='test_ray_her',
                run=train_ddpg_her,
                local_dir=OUT_DIR,
                config=dict(
                    env="FetchPickAndPlace-v1",
                    n_workers=2,
                    checkpoint_freq=10,
                )
            )
        )
    else:
        train_ddpg_her()


def train_with_mpi_only():

    def reporter(**kwargs):
        print(kwargs)

    config = dict(
        env="FetchPickAndPlace-v1",
        n_workers=5,
        n_epochs=2,
        checkpoint_freq=10,
        local_dir=f'{OUT_DIR}/simple_mpi'
    )

    train_ddpg_her(config, reporter)


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    train_with_mpi_only()
