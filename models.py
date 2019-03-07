import torch
import torch.nn as nn
import torch.nn.functional as F
from gym.spaces import Box, Dict

"""
the input x in both networks should be [o, g], where o is the observation and g is the goal.

"""


class ActorNetwork(nn.Module):
    def __init__(self, *, action_space: Box, observation_space: Dict, hidden_units: int):
        super(ActorNetwork, self).__init__()

        obs_size = observation_space.spaces['observation'].shape[0]
        goal_size = observation_space.spaces['desired_goal'].shape[0]

        self.fc1 = nn.Linear(obs_size + goal_size, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, hidden_units)
        self.action_out = nn.Linear(hidden_units, action_space.shape[0])
        self.max_action = action_space.high[0]

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions


class CriticNetwork(nn.Module):
    def __init__(self, *, action_space: Box, observation_space: Dict, hidden_units: int):
        super(CriticNetwork, self).__init__()

        obs_size = observation_space.spaces['observation'].shape[0]
        goal_size = observation_space.spaces['desired_goal'].shape[0]

        self.fc1 = nn.Linear(obs_size + goal_size + action_space.shape[0], hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, hidden_units)
        self.q_out = nn.Linear(hidden_units, 1)
        self.max_action = action_space.high[0]

    def forward(self, x, actions):
        x = torch.cat((x, actions / self.max_action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)

        return q_value
