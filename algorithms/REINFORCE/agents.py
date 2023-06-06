import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
	def __init__(self, input_dims, n_actions, hidden_dim=128, lr=0.001):
		super(PolicyNetwork, self).__init__()
		self.fc1 = nn.Linear(*input_dims, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, hidden_dim)
		self.fc3 = nn.Linear(hidden_dim, n_actions)
		
		self.optimizer = optim.Adam(self.parameters(), lr=lr)

		# send the whole object to operate on the gpu if available, else on the cpu
		self.to(self.device)

	@property
	def device(self):
		return T.device('cuda:0' if T.cuda.is_available() else 'cpu')

	def forward (self, state):
		'''
		Yields the preference function over the action space
		for the state given as input

		The probability distribution over the action can be defined by,
		e.g. taking the softmax of the output.

		'''
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)	# no activation whatever here for now
		return x


class PolicyGradientAgent():
	def __init__(self, input_dims, n_actions=4, hidden_dim=128, gamma=0.99, lr=0.001):
		self.gamma = gamma
		self.lr = lr
		self.n_actions = n_actions

		self.policy_net = PolicyNetwork(input_dims, n_actions, hidden_dim=hidden_dim, lr=self.lr)

		self.memory = []

		# why does he split the thing?
		self.reward_memory = []
		self.logprob_memory = []

	@property
	def device(self):
		return self.policy_net.device

	@property
	def optimizer(self):
		return self.policy_net.optimizer
	

	def choose_action (self, state):
		'''
		Action selection with the softmax 

		'''

		# first we need to convert the state from a numpy.ndarray to a 
		# torch.Tensor object. Note that the first index should always 
		# have dimension 1 (it must be seen as a row vector)
		state = T.Tensor(state).view(1,len(state))

		# send this state to the device we are using
		state = state.to(self.device)

		# calculate the preference function, which is the term in the exponent
		# in calculating the softmax (h function of Sutton&Barto)
		preference = self.policy_net.forward(state)

		# Calculate the soft-max over the preference
		# The dim=1 optional argument specifies over which 'axis' of the output
		# should the normalization be done.
		# The output of the forward pass is a (1, n_actions)-dim tensor,
		# so the normalization should be done on axis 1
		probs = F.softmax(preference, dim=1)

		# define a pytorch categorical distribution with the softmax probabilities
		distr = Categorical(probs)

		# sample the action from it
		action = distr.sample() # this is a pytorch tensor

		# calculate the logarithm of the policy evaluated at that action
		# and store it for the policy-gradient calculation
		logprob = distr.log_prob(action)
		self.logprob_memory.append(logprob)

		# since 'action' is a tensor, in order to return the corresponding
		# float (or array of floats for multidimensional actions) ne need
		# to use the 'item' method
		return action.item()

	def store_reward(self, reward):
		'''
		a utility function to store the reward in memory
		'''
		self.reward_memory.append(reward)

	def calculate_returns(self):
		'''
		Calculate the (discounted) return cumulated from all the time-steps
		in the episode
		'''
		G = np.zeros_like(self.reward_memory, dtype=np.float64)
		for t in range(len(self.reward_memory)):
			discount = 1.
			R = 0
			for k in range(t, len(self.reward_memory)):
				R += self.reward_memory[k] * discount
				discount *= self.gamma
			G[t] = R
		return G

	def update_policy(self):

		self.optimizer.zero_grad()

		G = self.calculate_returns()
		G = T.Tensor(G).to(self.device)

		# so, as long as 'loss' is a torch.Tensor, then it can
		# be calculated however we want, and doesn't necessarily
		# have to be a standard function or coded as a 
		# torch.nn.functional object?!?!?! INTERESTING
		loss = 0.
		for g, l in zip(G, self.logprob_memory):
			loss += - g * l

		# calculate the gradient by backpropagation
		loss.backward()

		# take a step with the optimizer
		self.optimizer.step()

		# reset the memory before starting a new episode
		self.reward_memory = []
		self.logprob_memory = []
