import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ..processes import OrnsteinUhlenbeck
from .networks import ActorNetwork, CriticNetwork
from ..memory import ReplayMemory



class DDPGAgent ():
    def __init__(self, input_dims, n_actions, tau=0.001, gamma=0.99,
                 lr_actor=1e-4, lr_critic=1e-3, weight_decay=1e-2, # optimizer
                 omega=0.2, D=0.01, # ornstein-uhlenbeck
                 fc1_dims=400, fc2_dims=300, # architecture
                 batch_size=64, buffer_size=1000000, # replay
                 filename="model", checkpoint_dir="checkpoints", # output
                 **kwargs):
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.alpha = lr_critic
        self.beta = lr_actor
        self.omega = omega
        self.D = D

        self.memory = ReplayMemory(self.input_dims, self.n_actions, buffer_size=buffer_size)

        self.noise = OrnsteinUhlenbeck(np.zeros(self.n_actions), omega=self.omega, D=self.D)

        self.actor = ActorNetwork(self.input_dims, self.n_actions, lr=self.beta,
                                  fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims,
                                  checkpoint_dir=checkpoint_dir,
                                  filename=f"{filename}_actor")
        self.critic = CriticNetwork(self.input_dims, self.n_actions, lr=self.alpha,
                                    fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims,
                                    checkpoint_dir=checkpoint_dir,
                                    filename=f"{filename}_critic",
                                    weight_decay=weight_decay)
        # targets are identical to the original instances
        # (target do not need optimization parameters, but only input, output
        # and hidden layer sizes)
        self.target_actor = ActorNetwork(self.input_dims, self.n_actions,
                                         fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims,
                                         checkpoint_dir=checkpoint_dir,
                                         filename=f"{filename}_target_actor")
        self.target_critic = CriticNetwork(self.input_dims, self.n_actions,
                                           fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims,
                                           checkpoint_dir=checkpoint_dir,
                                           filename=f"{filename}_target_critic")

        self.update_parameters(tau=1.)


    def choose_action(self, state, numpy=True):
        # If we have (batch) normalization layers, we need to keep track of
        # layer norm statistics when we do the training, because those are
        # quantities used in training the parameters of those layers.
        # Here, we want just to calculate the action given the state, and so we
        # don't need to do that.
        # In order to avoid calculating the norm statistics of the (batch)
        # normalization layers, we need to set the network in evaluation mode.
        # If no (batch) normalization is used, this is not necessary.
        self.actor.eval()

        # convert state to tensor and send to device
        state = T.tensor([state], dtype=T.float).to(self.actor.device)

        mu = self.actor.forward(state).to(self.actor.device)
        mu += T.tensor(self.noise(), dtype=T.float).to(self.actor.device)

        # at the end of the calculation we send back the actor network
        # to training mode
        self.actor.train()

        # 1. sum the noise to the deterministic action
        # 2. copy the result to cpu
        # 3. detach? perhaps simply saying that you don't want to calculate gradients of this?
        # 4. convert to numpy array (it will be of shape (1, n_actions), so take the only element in it)
        if numpy:
            return mu.cpu().detach().numpy()[0]
        else:
            return mu

    def save_models(self):
        print("--- saving checkpoints ---")
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        print("--- loading checkpoints ---")
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def store (self, state, action, reward, state_, done):
        self.memory.store(state, action, reward, state_, done)

    def learning_step(self):

        if self.memory.mem_counter < self.batch_size:
            return


        states, actions, rewards, states_, dones = self.memory.batch(self.batch_size)

        '''
        1. Tranform into tensors all arrays in the batch (the first dimension
           of each should have size batch_size)
        2. With the TARGET NETWORKS, Q', mu':
            a. compute deterministic action mu'(s_) given state_ with policy network
            b. compute Q'(s, mu'(s_))
        3. With the ONLINE NETWORKS:
            b. as above but with state before transition, and with the sampled 
               actions, Q(s,a)
        4. define the target for the online Q, y = r + gamma Q'(s_, mu'(s_))
        5. define the TD error:  delta = y - Q(s, a)
        6. Cumulate the losses:
            a. loss for critic = <delta**2>_batch
            b. loss for actor = - <Q(s, mu(s))>_batch
        7. backpropagate the losses
        8. perform the steps of the Adam optimizer
        '''
        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).view(-1,1).to(self.actor.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.actor.device)
        dones = T.tensor(dones).to(self.actor.device)

        mus_ = self.target_actor.forward(states_)
        values_ = self.target_critic.forward(states_, mus_)
        values_[dones] = 0.
        
        mus = self.actor.forward(states)
        values = self.critic.forward(states, actions)

        ys = rewards + self.gamma * values_ 

        self.critic.optimizer.zero_grad()
        loss_critic = F.mse_loss(ys, values)
        loss_critic.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        loss_actor = - T.mean(self.critic.forward(states, mus)) # (values_det)
        loss_actor.backward()
        self.actor.optimizer.step()

        self.update_parameters()


    def update_parameters(self, tau=None):

        if tau is None:
            tau = self.tau

        actor_pars = dict(self.actor.named_parameters())
        critic_pars = dict(self.critic.named_parameters())
        target_actor_pars = dict(self.target_actor.named_parameters())
        target_critic_pars = dict(self.target_critic.named_parameters())

        for key in actor_pars:
            actor_pars[key] = (1. - tau) * target_actor_pars[key].clone() \
                                     + tau * actor_pars[key].clone()

        for key in critic_pars:
            critic_pars[key] = (1. - tau) * target_critic_pars[key].clone() \
                                      + tau * critic_pars[key].clone()

        self.target_actor.load_state_dict(actor_pars)
        self.target_critic.load_state_dict(critic_pars)
        # if one wants to use batch normalization layers, instead of layer
        # normalization, one needs to specify the 'strict' option to False:
        # this copies also the batch norm statistics
        # self.target_actor.load_state_dict(target_actor_pars, strict=False)
        # self.target_actor.load_state_dict(target_actor_pars, strict=False)

    def __repr__(self):
        info = ''
        info += "DDPG agent networks configuration\n"
        info += "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\n"
        info += "Actor network\n"
        info += "-------------\n"
        info += f"Running on {self.actor.device}\n"
        info += f"{self.actor}\n\n"
        info += "Critic network\n"
        info += "-------------\n"
        info += f"Running on {self.critic.device}\n"
        info += f"{self.critic}\n"
        return info


class DDPGAgent_value_init(DDPGAgent):
    '''
    A DDPG agent that can initialize the critic parameters by fitting
    an existing sample of states, actions and values (if provided).

    '''
    @property
    def device(self):
        return self.actor.device

    def initialize(self, states, actions, values, n_epochs=5, batch_size=10):
        '''
        Runs a regression of values vs states using the value network

        '''
        states = states.reshape(-1, *self.input_dims)
        values = values.reshape(-1, 1)
        actions = values.reshape(-1, self.n_actions)

        print(states.shape)
        print(values.shape)
        print(actions.shape)

        assert states.shape[0] == values.shape[0], \
               "different number of states and values training points"
        self.pre_training_size = states.shape[0]

        # evaluate naive network
        with T.no_grad():
            _states = T.tensor(states, dtype=T.float).to(self.device)
            _values = T.tensor(values, dtype=T.float).to(self.device)
            _actions = self.actor.forward(_states)
            outputs = self.critic.forward(_states, _actions) # is this equivalent to net.forward(batch_X)?
            loss = T.nn.functional.mse_loss(outputs, _values)
            print(f'Start.    Loss: {loss}')

        for epoch in range(n_epochs):
            # shuffle indices
            ids = np.random.permutation(np.arange(self.pre_training_size))

            for i in range(0, self.pre_training_size, batch_size):
                # select a subset of the (shuffled) indices
                batch = ids[i:i+batch_size]
                # print(batch)
                # select corresponding states and values, and select actions based
                # on the initial random policy
                _states = T.tensor(states[batch], dtype=T.float).to(self.device)
                _values = T.tensor(values[batch], dtype=T.float).to(self.device)
                # _actions = T.tensor(actions[batch], dtype=T.float).to(self.device)

                with T.no_grad():
                    # select actions (deterministically, but according to random parameters)
                    _actions = self.actor.forward(_states)

                # perform a batch update (least square regression)
                self.critic.zero_grad()
                outputs = self.critic.forward(_states, _actions) # is this equivalent to net.forward(batch_X)?
                loss = T.nn.functional.mse_loss(outputs, _values)
                loss.backward()
                self.critic.optimizer.step()
            
            print(f'Epoch: {epoch+1}. Loss: {loss}')

        # copy the parameters in the target networks
        self.update_parameters(tau=1.)


    def test_initialize(self):
        raise NotImplementedError