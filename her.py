import numpy as np


class HerSampler:

    def __init__(self, replay_strategy, replay_k, reward_func=None, *, weight_sampling, weight_rewards):
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func
        self.weight_sampling = weight_sampling
        self.weight_rewards = weight_rewards

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions, *, get_weights_for_goals=None):

        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions

        # select which rollouts and which timesteps to be used
        if self.weight_sampling and callable(get_weights_for_goals):
            all_goals = episode_batch['g'][:, 0]
            weights = get_weights_for_goals(all_goals) # this may be expensive when buffer is large
            episode_idxs = np.random.choice(rollout_batch_size, p=weights, size=batch_size)
        else:
            episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}

        # her idx
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        # replace go with achieved goal
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag

        info = None
        if self.weight_rewards and callable(get_weights_for_goals):
            info = dict(weights=get_weights_for_goals(transitions['g']))

        # to get the params to re-compute reward
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], info), 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        return transitions
