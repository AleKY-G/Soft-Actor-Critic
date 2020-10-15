from networks import SoftQNetwork, PolicyNetwork
from replay_buffer import ReplayBuffer
import torch.optim as optim
import torch
from torch.distributions.normal import Normal
import numpy as np
import os

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class SAC_Agent:
    def __init__(self, env, batch_size=256, gamma=0.99, tau=0.005, actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4):
        #Environment
        self.env = env
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]        
        
        #Hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        #Entropy
        self.alpha = 1
        self.target_entropy = -np.prod(env.action_space.shape).item()  # heuristic value
        self.log_alpha = torch.zeros(1, requires_grad=True, device="cuda")
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        #Networks
        self.Q1 = SoftQNetwork(state_dim, action_dim).cuda()
        self.Q1_target = SoftQNetwork(state_dim, action_dim).cuda()
        self.Q1_target.load_state_dict(self.Q1.state_dict())
        self.Q1_optimizer = optim.Adam(self.Q1.parameters(), lr=critic_lr)

        self.Q2 = SoftQNetwork(state_dim, action_dim).cuda()
        self.Q2_target = SoftQNetwork(state_dim, action_dim).cuda()
        self.Q2_target.load_state_dict(self.Q2.state_dict())
        self.Q2_optimizer = optim.Adam(self.Q2.parameters(), lr=critic_lr)

        self.actor = PolicyNetwork(state_dim, action_dim).cuda()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        self.loss_function = torch.nn.MSELoss()
        self.replay_buffer = ReplayBuffer()

    def act(self, state, deterministic=True):
        state = torch.tensor(state, dtype=torch.float, device="cuda")
        mean, log_std = self.actor(state)  
        if(deterministic):
            action = torch.tanh(mean)
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
            z = normal.sample()
            action = torch.tanh(z)
        action = action.detach().cpu().numpy()
        return action

    def update(self, state, action, next_state, reward, done):

        self.replay_buffer.add_transition(state, action, next_state, reward, done)

        # Sample next batch and perform batch update: 
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = \
            self.replay_buffer.next_batch(self.batch_size)
        
        #Map to tensor
        batch_states = torch.tensor(batch_states, dtype=torch.float, device="cuda") #B,S_D
        batch_actions = torch.tensor(batch_actions, dtype=torch.float, device="cuda") #B,A_D
        batch_next_states = torch.tensor(batch_next_states, dtype=torch.float, device="cuda", requires_grad=False) #B,S_D
        batch_rewards = torch.tensor(batch_rewards, dtype=torch.float, device="cuda", requires_grad=False).unsqueeze(-1) #B,1
        batch_dones = torch.tensor(batch_dones, dtype=torch.uint8, device="cuda", requires_grad=False).unsqueeze(-1) #B,1

        #Policy evaluation
        with torch.no_grad():
            policy_actions, log_pi = self.actor.sample(batch_next_states)
            Q1_next_target = self.Q1_target(batch_next_states, policy_actions)
            Q2_next_target = self.Q2_target(batch_next_states, policy_actions)
            Q_next_target = torch.min(Q1_next_target, Q2_next_target)
            td_target = batch_rewards + (1 - batch_dones) * self.gamma * (Q_next_target - self.alpha * log_pi)

        Q1_value = self.Q1(batch_states, batch_actions)
        self.Q1_optimizer.zero_grad()
        loss = self.loss_function(Q1_value, td_target)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.Q1.parameters(), 1)
        self.Q1_optimizer.step()

        Q2_value = self.Q2(batch_states, batch_actions)
        self.Q2_optimizer.zero_grad()
        loss = self.loss_function(Q2_value, td_target)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.Q2.parameters(), 1)
        self.Q2_optimizer.step()

        #Policy improvement
        policy_actions, log_pi = self.actor.sample(batch_states)
        Q1_value = self.Q1(batch_states, policy_actions)
        Q2_value = self.Q2(batch_states, policy_actions)
        Q_value = torch.min(Q1_value, Q2_value)
        
        self.actor_optimizer.zero_grad()
        loss = (self.alpha * log_pi - Q_value).mean()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
        self.actor_optimizer.step()

        #Update entropy parameter 
        alpha_loss = (self.log_alpha * (-log_pi - self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()
        
        #Update target networks
        soft_update(self.Q1_target, self.Q1, self.tau)
        soft_update(self.Q2_target, self.Q2, self.tau)

    def save(self, file_name):
        torch.save({'actor_dict': self.actor.state_dict(),
                    'Q1_dict' : self.Q1.state_dict(),
                    'Q2_dict' : self.Q2.state_dict(),
                }, file_name)

    def load(self, file_name):
        if os.path.isfile(file_name):
            print("=> loading checkpoint... ")
            checkpoint = torch.load(file_name)
            self.actor.load_state_dict(checkpoint['actor_dict'])
            self.Q1.load_state_dict(checkpoint['Q1_dict'])
            self.Q2.load_state_dict(checkpoint['Q2_dict'])
            print("done !")
        else:
            print("no checkpoint found...")

    
