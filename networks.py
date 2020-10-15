import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units=256):
        super(SoftQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, 1)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units=256):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc_mean = nn.Linear(hidden_units, action_dim)
        self.fc_log_std = nn.Linear(hidden_units, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2) #Avoid -inf when std -> 0
        return mean, log_std
    
    def sample(self, state, epsilon=1e-6):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_pi = normal.log_prob(z) - torch.log(1 - action.square() + epsilon)
        log_pi = log_pi.sum(1, keepdim=True)
        return action, log_pi 

