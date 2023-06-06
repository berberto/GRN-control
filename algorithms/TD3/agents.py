import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .networks import ActorNetwork, CriticNetwork
from ..memory import ReplayMemory



class PerturbAction():
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, N=1):
        # always returns an array which has the mini-batch axis
        return np.random.normal(self.mu, self.sigma, size=(N, *self.mu.shape))


class TD3Agent ():
    def __init__(self, input_dims, n_actions, tau=0.005, gamma=0.99,
                 lr_critic=1e-3, lr_actor=1e-3, # optimizer learning rates
                 sigma_exp=0.1, sigma_smo=0.2, # exploration noise and smoothing noise
                 fc1_dims=400, fc2_dims=300, # architecture
                 weight_decay=0.0, # weight decay (L2 regularisation parameter)
                 batch_size=100, buffer_size=1000000, # replay
                 filename="model", checkpoint_dir="checkpoints", # output
                 update_delay=2, # number of learning steps in between target and actor
                 warmup = 1000, # number of timesteps to pick purely exploratory action
                 seed = 0
                ):

        T.manual_seed(seed)

        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.lr_critic = lr_critic
        self.lr_actor = lr_actor
        self.sigma_exp = sigma_exp
        self.sigma_smo = sigma_smo
        self.update_delay = update_delay
        self.update_counter = 0
        self.warmup_time = warmup
        self.warmup_counter = 0

        self.noise = PerturbAction(np.zeros(self.n_actions), sigma=self.sigma_exp)
        self.smoothing = PerturbAction(np.zeros(self.n_actions), sigma=self.sigma_smo)

        actor_options = {"lr": self.lr_actor,
                         "weight_decay": weight_decay,
                         "fc1_dims": self.fc1_dims,
                         "fc2_dims": self.fc2_dims,
                         "checkpoint_dir": checkpoint_dir}
        critic_options = {"lr": self.lr_critic,
                         "weight_decay": weight_decay,
                         "fc1_dims": self.fc1_dims,
                         "fc2_dims": self.fc2_dims,
                         "checkpoint_dir": checkpoint_dir}

        dimensions = (self.input_dims, self.n_actions)

        self.memory = ReplayMemory(*dimensions, buffer_size=buffer_size)
        
        self.actor = ActorNetwork(*dimensions, **actor_options, filename=f"{filename}_actor")
        self.critic_1 = CriticNetwork(*dimensions, **critic_options, filename=f"{filename}_critic_1")
        self.critic_2 = CriticNetwork(*dimensions, **critic_options, filename=f"{filename}_critic_2")

        self.target_actor = ActorNetwork(*dimensions, **actor_options, filename=f"{filename}_target_actor")
        self.target_critic_1 = CriticNetwork(*dimensions, **critic_options, filename=f"{filename}_target_critic_1")
        self.target_critic_2 = CriticNetwork(*dimensions, **critic_options, filename=f"{filename}_target_critic_2")

        self.update_parameters(tau=1.)

    @property
    def critic(self):
        return self.critic_1

    def _random_action (self):
        return T.tensor(self.noise(), dtype=T.float).to(self.actor.device)

    def choose_action(self, state, numpy=True):

        # if still in the warm-up phase, select a random action and 
        # increment the warm-up step counter
        if self.warmup_counter < self.warmup_time:
            mu = self._random_action()
            self.warmup_counter += 1
        # if not in the warm-up phase, select the action through the 
        # actor network and add some noise to it
        else:
            state = T.tensor(state, dtype=T.float).view(1,*self.input_dims).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)
        mu += self._random_action()
        mu = T.clip(mu, -1, 1)

        if numpy:
            return mu.cpu().detach().numpy()[0]
        else:
            return mu

    def save_models(self):
        print("--- saving checkpoints ---")
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        print("--- loading checkpoints ---")
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_2.load_checkpoint()
        self.warmup_time = 0

    def store (self, state, action, reward, state_, done):
        self.memory.store(state, action, reward, state_, done)

    def batch_to_tensor (self, states, actions, rewards, states_, dones):
        return (
                T.tensor(states, dtype=T.float).to(self.actor.device),
                T.tensor(actions, dtype=T.float).to(self.actor.device),
                T.tensor(rewards, dtype=T.float).view(-1,1).to(self.actor.device),
                T.tensor(states_, dtype=T.float).to(self.actor.device),
                T.tensor(dones).to(self.actor.device)
               )

    def learning_step(self, targets_and_actor=False):

        if self.memory.mem_counter < self.batch_size:
            return

        states, actions, rewards, states_, dones = self.batch_to_tensor(*self.memory.batch(self.batch_size))

        # perturb action that is used to calculate the smoothing of the target and clip
        smoothing = self.smoothing(N=self.batch_size)

        smoothing = T.clip(T.tensor(smoothing, dtype=T.float), -0.5, 0.5).to(self.actor.device)
        mus_ = self.target_actor.forward(states_)
        mus_ = T.clip(mus_ + smoothing, -1, 1)

        # target value being the minimum of the two target critics
        values1_ = self.target_critic_1.forward(states_, mus_)
        values2_ = self.target_critic_2.forward(states_, mus_)

        values1_[dones] = 0.0
        values2_[dones] = 0.0

        values_ = T.min(values1_, values2_)
        
        values1 = self.critic_1.forward(states, actions)
        values2 = self.critic_2.forward(states, actions)

        ys = rewards + self.gamma * values_ 

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        loss_critic = F.mse_loss(ys, values1) + F.mse_loss(ys, values2)
        loss_critic.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_counter += 1

        if self.update_counter == self.update_delay:
            # update actor and target networks
            mus = self.actor.forward(states)

            self.actor.optimizer.zero_grad()
            loss_actor = - T.mean(self.critic_1.forward(states, mus)) # (values_det)
            loss_actor.backward()
            self.actor.optimizer.step()

            self.update_parameters()
            self.update_counter = 0



    def update_parameters(self, tau=None):

        if tau is None:
            tau = self.tau

        actor_pars = dict(self.actor.named_parameters())
        critic_1_pars = dict(self.critic_1.named_parameters())
        critic_2_pars = dict(self.critic_2.named_parameters())
        target_actor_pars = dict(self.target_actor.named_parameters())
        target_critic_1_pars = dict(self.target_critic_1.named_parameters())
        target_critic_2_pars = dict(self.target_critic_2.named_parameters())

        # for pars, target in zip([critic_1_pars,        critic_2_pars,        actor_pars],
        #                         [target_critic_1_pars, target_critic_2_pars, target_actor_pars]):
        #     for key in pars:
        #         target[key] = (1. - tau) * target[key].clone() + tau * pars[key].clone()

        for name in critic_1_pars:
            critic_1_pars[name] = (1. - tau) * target_critic_1_pars[name].clone() + tau * critic_1_pars[name].clone()
        for name in critic_2_pars:
            critic_2_pars[name] = (1. - tau) * target_critic_2_pars[name].clone() + tau * critic_2_pars[name].clone()
        for name in actor_pars:
            actor_pars[name] = (1. - tau) * target_actor_pars[name].clone() + tau * actor_pars[name].clone()

        self.target_actor.load_state_dict(actor_pars)
        self.target_critic_1.load_state_dict(critic_1_pars)
        self.target_critic_2.load_state_dict(critic_2_pars)

    def __repr__(self):
        info = ''
        info += "TD3 agent networks configuration\n"
        info += "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\n"
        info += "Actor network\n"
        info += "-------------\n"
        info += f"{self.actor}\n"
        info += f"Running on {self.actor.device}\n\n"
        info += "Critic networks\n"
        info += "-------------\n"
        info += f"{self.critic_1}\n"
        info += f"Running on {self.critic_1.device}\n"
        return info


'''
    MULTIAGENT VERSION
'''
from ..memory import ReplayMemory_MultiAgent

class PerturbAction_MultiAgent (object):
    def __init__(self, n_agents, mu, sigma):
        self.n_agents = n_agents
        self.mu = mu
        self.sigma = sigma

    def __call__(self, N=1):
        # always returns an array which has the mini-batch axis
        return np.random.normal(self.mu, self.sigma, size=(N, self.n_agents, *self.mu.shape))


class TD3MultiAgent (TD3Agent):

    def __init__(self, n_agents, input_dims, n_actions, tau=0.005, gamma=0.99,
                 lr_critic=1e-3, lr_actor=1e-3, # optimizer learning rates
                 sigma_exp=0.1, sigma_smo=0.2, # exploration noise and smoothing noise
                 fc1_dims=400, fc2_dims=300, # architecture
                 weight_decay=0.0, # weight decay (L2 regularisation parameter)
                 batch_size=100, buffer_size=1000000, # replay
                 filename="model", checkpoint_dir="checkpoints", # output
                 update_delay=2, # number of learning steps in between target and actor
                 warmup = 1000, # number of timesteps to pick purely exploratory action
                 seed = 0
                ):
        
        T.manual_seed(seed)

        self.n_agents = n_agents
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.lr_critic = lr_critic
        self.lr_actor = lr_actor
        self.sigma_exp = sigma_exp
        self.sigma_smo = sigma_smo
        self.update_delay = update_delay
        self.update_counter = 0
        self.warmup_time = warmup
        self.warmup_counter = 0

        self.noise = PerturbAction_MultiAgent(self.n_agents, np.zeros(self.n_actions), sigma=self.sigma_exp)
        self.smoothing = PerturbAction_MultiAgent(self.n_agents, np.zeros(self.n_actions), sigma=self.sigma_smo)

        actor_options = {"lr": self.lr_actor,
                         "weight_decay": weight_decay,
                         "fc1_dims": self.fc1_dims,
                         "fc2_dims": self.fc2_dims,
                         "checkpoint_dir": checkpoint_dir}
        critic_options = {"lr": self.lr_critic,
                         "weight_decay": weight_decay,
                         "fc1_dims": self.fc1_dims,
                         "fc2_dims": self.fc2_dims,
                         "checkpoint_dir": checkpoint_dir}

        dimensions = (self.input_dims, self.n_actions)

        self.memory = ReplayMemory_MultiAgent(self.n_agents, *dimensions, buffer_size=buffer_size)
        
        self.actor = ActorNetwork(*dimensions, **actor_options, filename=f"{filename}_actor")
        self.critic_1 = CriticNetwork(*dimensions, **critic_options, filename=f"{filename}_critic_1")
        self.critic_2 = CriticNetwork(*dimensions, **critic_options, filename=f"{filename}_critic_2")

        self.target_actor = ActorNetwork(*dimensions, **actor_options, filename=f"{filename}_target_actor")
        self.target_critic_1 = CriticNetwork(*dimensions, **critic_options, filename=f"{filename}_target_critic_1")
        self.target_critic_2 = CriticNetwork(*dimensions, **critic_options, filename=f"{filename}_target_critic_2")

        self.update_parameters(tau=1.)

    def _random_action (self):
        return T.tensor(self.noise(), dtype=T.float).to(self.actor.device)

    def choose_action(self, state, numpy=True):

        # if still in the warm-up phase, select a random action and 
        # increment the warm-up step counter
        if self.warmup_counter < self.warmup_time:
            mu = self._random_action()
            self.warmup_counter += 1
        # if not in the warm-up phase, select the action through the 
        # actor network and add some noise to it
        else:
            state = T.tensor(state, dtype=T.float).view(1,-1,*self.input_dims).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)
        mu += self._random_action()
        mu = T.clip(mu, -1, 1)

        if numpy:
            return mu.cpu().detach().numpy()[0]
        else:
            return mu

    def batch_to_tensor (self, states, actions, rewards, states_, dones):
        return (
                T.tensor(states, dtype=T.float).to(self.actor.device),
                T.tensor(actions, dtype=T.float).to(self.actor.device),
                T.tensor(rewards, dtype=T.float).view(-1,self.n_agents, 1).to(self.actor.device),
                T.tensor(states_, dtype=T.float).to(self.actor.device),
                T.tensor(dones).to(self.actor.device)
               )
