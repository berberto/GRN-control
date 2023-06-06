import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class ActorCriticNetwork(nn.Module):
	'''
	Network providing features for simultaneous approximation of value and policy
	functions for an Actor-Critic algorithm.

	Inheriting from torch.nn.Module base class.

	The network has the input layer and the first hidden layer in common between 
	policy and value function approximators.

	Initialization arguments
	------------------------
	
	- input_dims: tuple
		Dimension of the input layer. It is the shape of the input array/tensor

	- n_actions: int
		Number of actions for discrete control tasks or dimension of action space
		for continuous control tasks

	- features_dims: int or list of int
		Dimension(s) of the hidden layer(s) which are shared between actor and
		critic networks.

	- actor_dims: int, list of int, or None
		Dimension(s) of the additional hidden layer(s) for the actor network.
		These are stacked after the shared layers with dimensions specified
		by features_dims, and followed by the output layer, with dimension
		equal to n_actions.
		If None (default), it is equivalent to passing an empty list, which
		means that the output of the last shared layer is directly mapped
		to the output layer. If int, it is equivalent to passing a list containing
		that integer.

	- critic_dims: int, list of int, or None
		Equivalent use to actor_dims, but for the critic network.

	- gamma: float
		exponential discount factor

	- lr: float
		learning rate for the Adam optimizer


	Methods
	-------
	- forward(self, state, value_only=False):
		input: the state, as a torch.Tensor with shape (1,) + input_dims
		returns: (preference, value), 2-tuple of torch.Tensor objects
			

	Other methods as in the torch.nn.Module base class

	'''
	def __init__(self, input_dims, n_actions, lr=0.005,
			features_dims=[128], actor_dims=None, critic_dims=None
			):
		super(ActorCriticNetwork, self).__init__()

		self.input_dims = input_dims
		self.n_actions = n_actions
		self.lr = lr

		self.features_dims = [features_dims] if isinstance(features_dims, int) else features_dims

		if actor_dims is None:
			self.actor_dims = []
		elif isinstance(actor_dims, int):
			self.actor_dims = [actor_dims]
		else:
			self.actor_dims = actor_dims
		try:
			self.actor_dims.append(n_actions)
		except:
			raise ValueError("invalid type for \"actor_dims\" (it must be a list of int)")

		if critic_dims is None:
			self.critic_dims = []
		elif isinstance(critic_dims, int):
			self.critic_dims = [critic_dims]
		else:
			self.critic_dims = critic_dims
		try:
			self.critic_dims.append(1)
		except:
			raise ValueError("invalid type for \"critic_dims\" (it must be a list of int)")

		self.features_layers = nn.ModuleList()
		self.actor_layers = nn.ModuleList()
		self.critic_layers = nn.ModuleList()

		# setup fully connected layers for the feature extraction
		self.features_layers.append(nn.Linear(*input_dims, self.features_dims[0]))
		for l in range(1,len(self.features_dims)):
			self.features_layers.append(nn.Linear(self.features_dims[l-1], self.features_dims[l]))
			
		# setup fully connected layers for the actor (policy network)
		self.actor_layers.append(nn.Linear(self.features_dims[-1], self.actor_dims[0]))
		for l in range(1,len(self.actor_dims)):
			self.actor_layers.append(nn.Linear(self.actor_dims[l-1], self.actor_dims[l]))

		# setup fully connected layers for the critic (value network)
		self.critic_layers.append(nn.Linear(self.features_dims[-1], self.critic_dims[0]))
		for l in range(1,len(self.critic_dims)):
			self.critic_layers.append(nn.Linear(self.critic_dims[l-1], self.critic_dims[l]))

		# inizialize an optimizer for all the parameters
		self.optimizer = optim.Adam(self.parameters(), lr=lr)

		# send the whole object to operate on the gpu if available, else on the cpu
		self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
		self.to(self.device)

		self.__debug_info()


	def __debug_info(self):
		print(f"Performing calculations on {self.device}")
		# debugging info for layers
		for group, label in zip([self.features_layers, self.actor_layers, self.critic_layers], [
								'feature', 'actor', 'critic']):
			for i, l in enumerate(group):
				print(f"{label}\tlayer {i} ", l, "\tpars: ", list(map(lambda x: x.shape, list(l.parameters()))))


	def forward(self, state): #, value_only=False):
		'''
		Returns a 2-tuple with the preference vector over actions (or 
		action dimensions -- treated differently in choose_action) and
		the value function.

		Preferences and value are returned as torch.nn.Tensor objects

		If the optional argument 'value_only' is set to True, then only
		the value is calculated (the default value is False, for which
		both preferences and value are returned)

		'''

		# forward pass through common layers to extract the features
		fts = state
		for layer in self.features_layers:
			fts = F.relu(layer(fts))

		# forward pass through critic layers, to get the value of the function
		x = fts
		for layer in self.critic_layers[:-1]:
			x = F.relu(layer(x))
		value = self.critic_layers[-1](x)

		# forward pass through actor layers, to get the preference vector (policy)
		x = fts
		for layer in self.actor_layers[:-1]:
			x = F.relu(layer(x))
		prefs = self.actor_layers[-1](x)

		return prefs, value




class ActorCriticAgent(ActorCriticNetwork):
	'''
	Defines the behaviour of an Actor-Critic agent.

	Class derived from 'ActorCriticNetwork'

	Initialization arguments
	------------------------

	- input_dims: tuple
		dimension of the input layer. It is the shape of the input array/tensor

	- n_actions: int
		number of actions for discrete control tasks or dimension of action space
		for continuous control tasks

	- gamma: float
		exponential discount factor


	Methods
	-------
	- choose_action(state):
		takes as input an array-like object 'state', and returns a random
		action extracted from the softmax distribution over the output
		of the actor part of the ActorCriticNetwork.

	- learning_step(state, reward, state_, done):
		performs the one-step (TD(0)) online update of the actor-critic
		algoritm.

	For other methods and arguments, see the ActorCriticNetwork class.

	'''
	def __init__(self, input_dims, n_actions, gamma=0.99, **kwargs):
					# features_dims=128, actor_dims=128, critic_dims=128, \
					#  lr=0.001, ):
		super(ActorCriticAgent,self).__init__(input_dims, n_actions, **kwargs)
								# features_dims=features_dims, actor_dims=actor_dims,
								# critic_dims=critic_dims, lr=lr, 
		self.gamma = gamma
		self._log_prob = None
		self._value = None


	def choose_action (self, state):
		'''
		Action selection with the softmax 

		'''
		# first we need to convert the state from a numpy.ndarray to a 
		# torch.Tensor object. Note that the first index should always 
		# have dimension 1 (it must be seen as a row vector)
		state = T.tensor([state], dtype=T.float).to(self.device)


		# Calculate the soft-max over the preference
		# The dim=1 optional argument specifies over which 'axis' of the output
		# should the normalization be done.
		# The output of the forward pass is a (1, n_actions)-dim tensor,
		# so the normalization should be done on axis 1
		probs, self._value = self.forward(state)
		probs = F.softmax(probs, dim=1)

		# define a pytorch categorical distribution with the softmax probabilities
		distr = Categorical(probs)

		# sample the action from it
		action = distr.sample() # this is a pytorch tensor

		# calculate the logarithm of the policy evaluated at that action
		# and store it for the policy-gradient calculation
		self._log_prob = distr.log_prob(action)

		# since 'action' is a tensor, in order to return the corresponding
		# float (or array of floats for multidimensional actions) ne need
		# to use the 'item' method
		return action.item()

	def learning_step(self, state, reward, state_, done):
		'''
		Compute a TD step for the actor-critic algorithm

		Input arguments
		---------------
		- state: array-like
			current state (unused)
		- reward: float
			reward received in the transition
		- state_: array-like
			state after the transition
		- done: bool
			whether the episode has terminated

		'''

		# reset the gradients of both optimizers
		self.optimizer.zero_grad()

		# state = T.tensor([state], dtype=T.float).to(self.device)
		reward = T.tensor(reward, dtype=T.float).to(self.device)
		state_ = T.tensor([state_], dtype=T.float).to(self.device)

		# _, value = self.forward(state)
		_, value_ = self.forward(state_)

		# (1 - int(done)) sets to 0 the target value for terminal states
		# self._value stored at choose_action step
		delta = reward + self.gamma * value_ * (1 - int(done)) - self._value
		
		# the critic minimizes the square norm of the TD error
		loss_critic = delta**2

		# the actor maximizes (minimizes the negative of) delta * log(pi(a|s))
		# BUT ISN'T THE POLICY GRADIENT TRYING TO MAXIMIZE LOG(PI(A|s)) ONLY,
		# WITH DELTA BEING A MULTIPLICATIVE CONSTANT IN FRONT OF THAT GRAD?
		# (self._log_prob stored at choose_action step)
		loss_actor = - self._log_prob * delta

		# calculate the gradient by backpropagation
		(loss_actor + loss_critic).backward()

		# take a step with the optimizer
		self.optimizer.step()
