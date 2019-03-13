import pickle
import copy
from time import sleep

from mpi4py import MPI
import pandas as pd
import numpy as np
import torch


def demonstrations_from_agent(env, agent, *, n, output_path=None, render=False, skip_episode=None):

    data = {k: [] for k in ['mb_obs', 'mb_ag', 'mb_g', 'mb_actions', 'mb_sim_states']}
    while len(data['mb_obs']) < n:

        ep_data = {k: [] for k in ['obs', 'ag', 'g', 'actions', 'sim_states']}
        done = False
        success = False

        obs = env.reset()
        goal = obs['desired_goal'].copy()
        if callable(skip_episode) and skip_episode(obs):
            # episode not interesting enough => skip
            continue

        while not done:
            action = agent.predict(obs)

            ep_data['obs'].append(obs['observation'].copy())
            ep_data['ag'].append(obs['achieved_goal'].copy())
            ep_data['g'].append(goal.copy())
            ep_data['actions'].append(action.copy())
            ep_data['sim_states'].append(copy.deepcopy(env.unwrapped.sim.get_state()))

            obs, reward, done, info = env.step(action)
            success = info['is_success'] and done
        ep_data['obs'].append(obs['observation'].copy())
        ep_data['ag'].append(obs['achieved_goal'].copy())
        ep_data['sim_states'].append(copy.deepcopy(env.unwrapped.sim.get_state()))

        if success:
            for k in ep_data.keys():
                data['mb_' + k].append(ep_data[k])
            print(f'{len(data["mb_obs"])}/{n} demonstrations recorded.')

            if render:
                for s in ep_data['sim_states']:
                    env.unwrapped.sim.set_state(copy.deepcopy(s))
                    env.unwrapped.sim.forward()
                    env.render()
                    sleep(0.01)

    for k in data.keys():
        if 'sim' in k:
            continue
        data[k] = np.array(data[k])

    if output_path is not None:
        pickle.dump(data, open(output_path, 'wb'))
    return data


def convert_baselines_demonstrations(file_path, *, output_path=None, max_ep_len):
    orig_data = np.load(file_path)
    orig_data_obs = orig_data['obs']
    orig_data_acs = orig_data['acs']

    data = dict(
        mb_obs=[[step['observation'] for step in ep[:(max_ep_len+1)]] for ep in orig_data_obs],
        mb_ag=[[step['achieved_goal'] for step in ep[:(max_ep_len+1)]] for ep in orig_data_obs],
        mb_g=[[step['desired_goal'] for step in ep[:max_ep_len]] for ep in orig_data_obs],
        mb_actions=[ep[:max_ep_len] for ep in orig_data_acs]
    )

    for k in data.keys():
        data[k] = np.asarray(data[k])

    if output_path is not None:
        pickle.dump(data, open(output_path, 'wb'))
    return data


class StdoutReporter:

    def __init__(self, *, training_iter_key='training_iteration'):
        self._training_iter_key = training_iter_key

    def __call__(self, **kwargs):
        iter_i = kwargs.get(self._training_iter_key, '?')
        print('\n'+'#'*40)
        print(f'Iteration {iter_i}')
        print('#'*40)
        for k, v in kwargs.items():
            print(f'{k}: {v}')


class TensorboardReporter:

    def __init__(self, *, log_dir, training_iter_key='training_iteration', tags_prefix='data'):
        from tensorboardX import SummaryWriter
        self._writer = SummaryWriter(log_dir=log_dir)
        self._training_iter_key = training_iter_key
        self._tags_prefix = tags_prefix

    def __call__(self, **kwargs):
        iter_i = kwargs[self._training_iter_key]
        for k, v in kwargs.items():
            if self._tags_prefix is not None:
                k = f'{self._tags_prefix}/{k}'
            self._writer.add_scalar(k, v, iter_i)


class PandasReporter:

    def __init__(self, *, log_dir):
        self._log_path = f'{log_dir}/progress.csv'
        self._progress = []

    @property
    def data_frame(self):
        return pd.DataFrame(self._progress)

    def __call__(self, **kwargs):
        self._progress.append(kwargs)
        self.data_frame.to_csv(self._log_path)


class MultiReporter:

    def __init__(self, *, log_dir, tb=True, stdout=True, csv=True):
        reporters = []
        if tb:
            reporters.append(TensorboardReporter(log_dir=log_dir))
        if stdout:
            reporters.append(StdoutReporter())
        if csv:
            reporters.append(PandasReporter(log_dir=log_dir))
        self._reporters = reporters

    def __call__(self, **kwargs):
        for r in self._reporters:
            r(**kwargs)


# sync_networks across the different cores
def sync_networks(network):
    """
    netowrk is the network you want to sync

    """
    comm = MPI.COMM_WORLD
    flat_params, params_shape = _get_flat_params(network)
    comm.Bcast(flat_params, root=0)
    # set the flat params back to the network
    _set_flat_params(network, params_shape, flat_params)


# get the flat params from the network
def _get_flat_params(network):
    param_shape = {}
    flat_params = None
    for key_name, value in network.named_parameters():
        param_shape[key_name] = value.detach().numpy().shape
        if flat_params is None:
            flat_params = value.detach().numpy().flatten()
        else:
            flat_params = np.append(flat_params, value.detach().numpy().flatten())
    return flat_params, param_shape


# set the params from the network
def _set_flat_params(network, params_shape, params):
    pointer = 0
    for key_name, values in network.named_parameters():
        # get the length of the parameters
        len_param = np.prod(params_shape[key_name])
        copy_params = params[pointer:pointer + len_param].reshape(params_shape[key_name])
        copy_params = torch.tensor(copy_params)
        # copy the params
        values.data.copy_(copy_params.data)
        # update the pointer
        pointer += len_param


# sync the networks
def sync_grads(network):
    flat_grads, grads_shape = _get_flat_grads(network)
    comm = MPI.COMM_WORLD
    global_grads = np.zeros_like(flat_grads)
    comm.Allreduce(flat_grads, global_grads, op=MPI.SUM)
    _set_flat_grads(network, grads_shape, global_grads)


def _set_flat_grads(network, grads_shape, flat_grads):
    pointer = 0
    for key_name, value in network.named_parameters():
        len_grads = np.prod(grads_shape[key_name])
        copy_grads = flat_grads[pointer:pointer + len_grads].reshape(grads_shape[key_name])
        copy_grads = torch.tensor(copy_grads)
        # copy the grads
        value.grad.data.copy_(copy_grads.data)
        pointer += len_grads


def _get_flat_grads(network):
    grads_shape = {}
    flat_grads = None
    for key_name, value in network.named_parameters():
        grads_shape[key_name] = value.grad.data.cpu().numpy().shape
        if flat_grads is None:
            flat_grads = value.grad.data.cpu().numpy().flatten()
        else:
            flat_grads = np.append(flat_grads, value.grad.data.cpu().numpy().flatten())
    return flat_grads, grads_shape
