# Actor-Critic algorithms

The codes are written following the course by [Phil Tabor](https://github.com/philtabor), *Modern Reinforcement Learning: Actor-Critic algorithms* (on Udemy).
[Here](https://github.com/philtabor/Actor-Critic-Methods-Paper-To-Code) the codes from the instructor himself.


## REINFORCE (Policy Gradient)

**Codes**

- [agent](REINFORCE/agents.py)
- [main loop](../examples/lunarlander_REINFORCE.py)

REINFORCE uses a parametrized policy function.
It then performs MC estimates of the value function and combines them with the gradient of the policy (calculated analytically, or by automatic differentiation) through an analytical result -the *policy gradient theorem*- yielding an estimate of the gradient of the objective function with respect to the parameters of the policy.

It only works for episodic tasks!

---

## One-step actor-critic

**Codes**

- [agent](AC/agents.py)
- [main loop](../examples/lunarlander_actor_critic.py)

Actor-critic algorithms use both a parametrized policy (the *actor*) and an estimate of the (state or state-action) value function (the *critic*) and simultaneously learn both.
They use the result from the policy-gradient theorem to update the policy parameters, and use the value function as a baseline to reduce the variance of the estimate of the gradient.

Neural networks can be used as function approximators for the value function, as well as models for parametrized policies.
This implementation uses one single feed-forward neural network for both the actor and the critic:
- an initial set of layers is in common between the actor and the critic, serving as "feature extraction" part of the function;
- additional layers are stacked separately on top of these, to perform specific computations and mapping the extracted features into either the action or the value.
In principle one could use two entirely separate networks, but learning is harder.

---

## Deep Deterministic Policy Gradient

**Codes**

- [experience replay buffer](memory.py)
- [gaussian processes](processes.py), including the Ornstein-Uhlenbeck process
- [actor and critic networks](DDPG/networks.py)
- [agent](DDPG/agents.py)
- [main loop](../examples/lunarlander_ddpg.py)

**Original Paper**
> Lillicrap, T. P., et al. (2019). Continuous control with deep reinforcement learning. *ICLR conf.*, 2016 [https://arxiv.org/abs/1509.02971](https://arxiv.org/abs/1509.02971)

It is an actor-critic algorithm that learns to perform continuous control tasks. It combines:
- experience replay and batch updates to stabilize learning
- two deep feed-forward networks (online and target) for both policy and *state-action* value function, so 4 networks in total.
- "soft" (that is, slow integration) target updates
- a Gaussian colored noise (Ornstein-Uhlenbeck) for action exploration

It shares some features with Q-learning in that it relies on an **on-line**, **off-policy** estimate of the state-action value function Q.
The policy used to define the target value (meaning the one in the TD error) is the *deterministic* one, represented by the target actor, while the policy used to generate actions is a stochastic one, defined by the selection of a deterministic action by the online actor network and the addition of a colored noise.

It borrows from DQN
- experience replay to stabilize learning by removing correlations between transitions in the MDP,
- soft updates to stabilize learning by preventing large jumps of the target function.

---

## Twin Delayed Deep Deterministic (TD3) Actor Critic

**Codes**

- [experience replay buffer](memory.py)
- [actor and critic networks](TD3/networks.py)
- [agent](TD3/agents.py)
- [training and testing](TD3/main.py) (for environments with and without reward shaping)
- [example](../examples/td3.py)

**Paper**
> S. Fujimoto, H. van Hoof, and D. Meger (2018), Addressing Function Approximation Error in Actor-Critic Methods, arXiv[1802.09477](https://arxiv.org/abs/1802.09477v3) 

TD3 uses two critic (hence *twin*) networks to better estimate the value function Q.
In standard Q-learning, the value of the state after the transition is taken to be the maximum over all actions of the Q function evaluated at that state, by boostrapping.
This is a problem that is present also in actor-critic algorithms like DDPG, where the "maximization over actions" is implicit in the policy-gradient formula.
This typically leads to an overestimation of the value (as demonstrated in the paper), and therefore to sub-optimal policies.

TD3 introduces two Q-networks for on-line learning, and two Q-networks to provide target values in the TD error minimization.
The target is calculated by replacing the value at the next state as the minimum between the values provided by the two target networks.
The two online Q networks learn indepentently, they use the same target at every update, and only one of them is used in the policy-gradient formula.
