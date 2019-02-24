import glob
import itertools as it
import os
import pickle
from datetime import datetime

import torch
import numpy as np
from mpi4py import MPI

from models import ActorNetwork, CriticNetwork
from utils import sync_networks, sync_grads
from replay_buffer import ReplayBuffer
from normalizer import Normalizer
from her import HerSampler


def get_env_params(env):
    obs = env.reset()
    return {
        'obs': obs['observation'].shape[0],
        'goal': obs['desired_goal'].shape[0],
        'action': env.action_space.shape[0],
        'action_max': env.action_space.high[0],
        'max_timesteps': env._max_episode_steps
    }


class DdpgHer(object):

    _default_config = {
        'n_epochs': 50,
        'n_cycles': 50,
        'n_batches': 40,
        'checkpoint_freq': 5,
        'seed': 123,
        'num_workers': 1,
        'replay_strategy': 'future',
        'clip_return': 50.,
        'noise_eps': 0.2,
        'random_eps': 0.3,
        'buffer_size': int(1e6),
        'replay_k': 4,
        'clip_obs': 200.,
        'batch_size': 256,
        'gamma': 0.98,
        'action_l2': 1.,
        'lr_actor': 0.001,
        'lr_critic': 0.001,
        'polyak': 0.95,
        'n_test_rollouts': 10,
        'clip_range': 5.,
        'demo_length': 20,
        'local_dir': None,
        'cuda': None,
        'rollouts_per_worker': 2,
    }

    def __init__(self, env, config, reporter=None):
        super(DdpgHer).__init__()

        self.env = env
        self.config = {**DdpgHer._default_config, **config}
        a_space, obs_space = self.env.action_space, self.env.observation_space
        obs_size = obs_space.spaces['observation'].shape[0]
        goal_size = obs_space.spaces['desired_goal'].shape[0]
        self.env_params = get_env_params(self.env)
        self.reporter = reporter

        if self.config['cuda'] is None:
            self.config['cuda'] = torch.cuda.is_available()

        if self.config['cuda']:
            n_gpus = torch.cuda.device_count()
            assert n_gpus > 0
            n_workers = MPI.COMM_WORLD.size
            rank = MPI.COMM_WORLD.rank
            w_per_gpu = int(np.ceil(n_workers / n_gpus))
            gpu_i = rank // w_per_gpu
            print(f'Worker with rank {rank} assigned GPU {gpu_i}.')
            torch.cuda.set_device(gpu_i)

        # create the network
        self.actor_network = ActorNetwork(action_space=a_space, observation_space=obs_space)
        self.critic_network = CriticNetwork(action_space=a_space, observation_space=obs_space)
        # sync the networks across the cpus
        sync_networks(self.actor_network)
        sync_networks(self.critic_network)
        # build up the target network
        self.actor_target_network = ActorNetwork(action_space=a_space, observation_space=obs_space)
        self.critic_target_network = CriticNetwork(action_space=a_space, observation_space=obs_space)
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        # if use gpu
        if self.config['cuda']:
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()
        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.config['lr_actor'])
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.config['lr_critic'])
        # her sampler
        self.her_module = HerSampler(self.config['replay_strategy'], self.config['replay_k'], self.env.compute_reward)
        # create the replay buffer
        self.buffer = ReplayBuffer(self.env_params, self.config['buffer_size'], self.her_module.sample_her_transitions)
        # create the normalizer
        self.o_norm = Normalizer(size=obs_size, default_clip_range=self.config['clip_range'])
        self.g_norm = Normalizer(size=goal_size, default_clip_range=self.config['clip_range'])
        self._trained = False

    def _training_step(self):
        rollout_times = []
        update_times = []
        taken_steps = 0
        step_tic = datetime.now()
        for _ in range(self.config['n_cycles']):
            mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
            for _ in range(self.config["rollouts_per_worker"]):
                tic = datetime.now()
                # reset the rollouts
                ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
                # reset the environment
                observation = self.env.reset()
                obs = observation['observation']
                ag = observation['achieved_goal']
                g = observation['desired_goal']
                # start to collect samples
                for t in range(self.env_params['max_timesteps']):
                    with torch.no_grad():
                        input_tensor = self._preproc_inputs(obs, g)
                        pi = self.actor_network(input_tensor)
                        action = self._select_actions(pi)
                    # feed the actions into the environment
                    observation_new, _, _, info = self.env.step(action)
                    taken_steps += 1
                    obs_new = observation_new['observation']
                    ag_new = observation_new['achieved_goal']
                    # append rollouts
                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    ep_g.append(g.copy())
                    ep_actions.append(action.copy())
                    # re-assign the observation
                    obs = obs_new
                    ag = ag_new
                ep_obs.append(obs.copy())
                ep_ag.append(ag.copy())
                mb_obs.append(ep_obs)
                mb_ag.append(ep_ag)
                mb_g.append(ep_g)
                mb_actions.append(ep_actions)
                rollout_times.append((datetime.now() - tic).total_seconds())

            # convert them into arrays
            mb_obs = np.array(mb_obs)
            mb_ag = np.array(mb_ag)
            mb_g = np.array(mb_g)
            mb_actions = np.array(mb_actions)
            # store the episodes
            self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
            self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])

            tic = datetime.now()
            # train the network
            for _ in range(self.config['n_batches']):
                self._update_network()
            # soft update
            self._soft_update_target_network(self.actor_target_network, self.actor_network)
            self._soft_update_target_network(self.critic_target_network, self.critic_network)
            update_times.append((datetime.now() - tic).total_seconds())
        step_time = (datetime.now() - step_tic).total_seconds()

        tic = datetime.now()
        success_rate = self._eval_agent()
        eval_time = (datetime.now() - tic).total_seconds()

        return {
            "test_success_rate": success_rate,
            "avg_rollout_time": np.mean(rollout_times),
            "avg_network_update_time": np.mean(update_times),
            "evaluation_time": eval_time,
            "step_time": step_time,
            "env_steps": taken_steps,
        }

    def save_checkpoint(self, epoch=0):
        local_dir = self.config.get('local_dir')
        if local_dir is not None:
            local_dir = local_dir + '/checkpoints'
            os.makedirs(local_dir, exist_ok=True)
            model_path = f'{local_dir}/model_{epoch}.pt'
            status_path = f'{local_dir}/status_{epoch}.pkl'
            torch.save([
                self.o_norm.mean,
                self.o_norm.std,
                self.g_norm.mean,
                self.g_norm.std,
                self.actor_network.state_dict()
            ], model_path)
            with open(status_path, 'wb') as f:
                pickle.dump(dict(config=self.config), f)

    @staticmethod
    def load(env, local_dir, epoch=None):
        epoch = epoch or '*[0-9]'
        models = glob.glob(f'{local_dir}/model_{epoch}.pt')
        assert len(models) > 0, "No checkpoints found!"

        model_path = sorted(models, key=os.path.getmtime)[-1]
        epoch = model_path.split("_")[-1].split(".")[0]
        status_path = f'{local_dir}/status_{epoch}.pkl'

        with open(status_path, 'rb') as f:
            status = pickle.load(f)
        agent = DdpgHer(env, status['config'])
        agent._trained = True

        o_mean, o_std, g_mean, g_std, actor_state = torch.load(model_path, map_location=lambda storage, loc: storage)

        agent.o_norm.mean = o_mean
        agent.o_norm.std = o_std
        agent.g_norm.mean = g_mean
        agent.g_norm.std = g_std

        agent.actor_network.load_state_dict(actor_state)
        agent.actor_network.eval()
        print(f'Loaded model for epoch {epoch}.')
        return agent

    def predict(self, obs):
        if not self._trained:
            raise RuntimeError
        g = obs['desired_goal']
        obs = obs['observation']
        inputs = self._preproc_inputs(obs, g)
        with torch.no_grad():
            pi = self.actor_network(inputs)
        action = pi.detach().numpy().squeeze()
        return action

    def train(self):
        if self._trained:
            raise RuntimeError
        tic = datetime.now()
        n_epochs = self.config.get('n_epochs')
        saved_checkpoints = 0

        for iter_i in it.count():
            if n_epochs is not None and iter_i >= n_epochs:
                break
            res = self._training_step()

            if MPI.COMM_WORLD.Get_rank() == 0:
                if (iter_i + 1) % self.config['checkpoint_freq'] == 0:
                    self.save_checkpoint(epoch=(iter_i + 1))
                    saved_checkpoints += 1
                if callable(self.reporter):
                    self.reporter(**{
                        **res,
                        "training_iteration": iter_i + 1,
                        "total_time": (datetime.now() - tic).total_seconds(),
                        "checkpoints": saved_checkpoints
                    })

    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.config['cuda']:
            inputs = inputs.cuda()
        return inputs
    
    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
        action += self.config['noise_eps'] * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        random_actions = np.random.uniform(
            low=-self.env_params['action_max'],
            high=self.env_params['action_max'],
            size=self.env_params['action']
        )
        # choose if use the random actions
        action += np.random.binomial(1, self.config['random_eps'], 1)[0] * (random_actions - action)
        return action

    # update the normalizer
    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {
            'obs': mb_obs,
            'ag': mb_ag,
            'g': mb_g,
            'actions': mb_actions,
            'obs_next': mb_obs_next,
            'ag_next': mb_ag_next,
        }
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.config['clip_obs'], self.config['clip_obs'])
        g = np.clip(g, -self.config['clip_obs'], self.config['clip_obs'])
        return o, g

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.config['polyak']) * param.data + self.config['polyak'] * target_param.data)

    # update the network
    def _update_network(self):
        # sample the episodes
        transitions = self.buffer.sample(self.config['batch_size'])
        # pre-process the observation and goal
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
        # start to do the update
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32) 
        if self.config['cuda']:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()
        # calculate the target Q value function
        with torch.no_grad():
            # do the normalization
            # concatenate the stuffs
            actions_next = self.actor_target_network(inputs_next_norm_tensor)
            q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.config['gamma'] * q_next_value
            target_q_value = target_q_value.detach()
            # clip the q value
            clip_return = 1 / (1 - self.config['gamma'])
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)
        # the q loss
        real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        # the actor loss
        actions_real = self.actor_network(inputs_norm_tensor)
        actor_loss = -self.critic_network(inputs_norm_tensor, actions_real).mean()
        actor_loss += self.config['action_l2'] * (actions_real / self.env_params['action_max']).pow(2).mean()
        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor_network)
        self.actor_optim.step()
        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic_network)
        self.critic_optim.step()

    # do the evaluation
    def _eval_agent(self):
        total_success_rate = []
        for _ in range(self.config['n_test_rollouts']):
            per_success_rate = []
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    pi = self.actor_network(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
                observation_new, _, _, info = self.env.step(actions)
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                per_success_rate.append(info['is_success'])
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size()
