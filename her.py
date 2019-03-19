import numpy as np
from scipy.special import softmax


class HerSampler:

    def __init__(self, replay_strategy, replay_k, reward_func=None, *, weight_sampling=False, archer_params=None):
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func
        self.weight_sampling = weight_sampling
        self.archer_params = archer_params

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions, *, get_info_for_goals=None):

        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions

        # sample rollouts
        if self.weight_sampling and callable(get_info_for_goals):

            info = get_info_for_goals(episode_batch['ag'])
            x_success, tot_success, x_visited, tot_visited, x_observed, tot_observed = info
            weights = (x_visited / (tot_visited + 1e-9)) * (x_observed / (tot_observed + 1e-9)) * (1. / (1. + x_success))
            weights = np.log(weights + 1e-100).sum(axis=-1)
            weights = softmax(weights)
            weights /= weights.sum()

            episode_idxs = np.random.choice(rollout_batch_size, p=weights, size=batch_size)
        else:
            episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)

        # sample timesteps
        t_samples = np.random.randint(T, size=batch_size)

        # get transitions corresponding to the sampled rollouts and timesteps
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}

        # sample HER indexes
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)[0]
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        # replace goals with achieved goals
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag

        # re-compute the rewards for the modified goals
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)

        if self.archer_params is not None:
            lambda_r = self.archer_params['lambda_r']
            lambda_h = self.archer_params['lambda_h']
            not_her_indexes = np.ones(batch_size, dtype=bool)
            not_her_indexes[her_indexes] = False
            transitions['r'][her_indexes] *= lambda_h
            transitions['r'][not_her_indexes] *= lambda_r

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}
        return transitions
