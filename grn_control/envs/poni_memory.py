import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from os import path
from .cost_functions import eps_relu_target, eps_relu_target_pattern
from .cost_functions import quadratic_target, quadratic_target_pattern
from .poni_aux import f_poni, f_signal, f_memory, poni_target, poni_colors_array

from .poni_signal import PONINetwork_Diffusion_Pattern


# PONI-pattern-v4
class PONINetwork_Diffusion_Memory(PONINetwork_Diffusion_Pattern):

    def __init__ (self, d_memory=None):

        # number of memory variables
        if d_memory is None:
            d_memory = 1
        self.d_memory = d_memory

        self.tau_mem = 1.  # time scale for memory variables
        self.d_state = 4
        self.d_signal = 1
        self.d_control = 2
        self.d_action = self.d_control + self.d_memory  # action = control & prod rates memory vars

        self.noise = 5e-3  # None  # noise strength

        # restart bounds
        self.restart_low =  np.concatenate(([0.9, 0.0, 0.0, 0.9], [0.0], np.zeros(self.d_memory)))
        self.restart_high = self.restart_low + 0.1

        # bounds state space
        self.high = 1.2*np.ones(self.d_state + self.d_signal + self.d_memory, dtype=float)
        self.high[-(self.d_memory+self.d_signal):] = 10.  # high bound for signal & memory variables
        self.low = np.zeros_like(self.high, dtype=float)

        # bounds action space
        self.max_u = np.zeros(self.d_action, dtype=float)
        self.max_u[:2] = 2.
        self.max_u[-self.d_memory:] = 1.
        self.min_u = np.zeros(self.d_action, dtype=float)

        super(PONINetwork_Diffusion_Memory, self).__init__()

    def get_dims (self):
        '''
        Returns arrays with integers corresponding to the dimensions in the 
        state vector for "state", "signal" and "memory" variables
        '''
        return (np.arange(self.d_state),
                np.arange(self.d_signal) + self.d_state,
                np.arange(self.d_memory) + self.d_state + self.d_signal
               )

    def _split_state_vector (self, x):
        dims_state, dims_signal, dims_memory = self.get_dims()
        state = x[dims_state, :]
        signal = x[dims_signal, :]
        memory = x[dims_memory, :]
        return state, signal, memory

    def reset (self, state=None):
        '''
        Returns the OBSERVATION that is given in a random initial state
        '''
        self.steps_beyond_done = None
        self.prev_shaping = None
        self.time = 0.
        low = self.restart_low
        high = self.restart_high
        if state is not None:
            assert state.shape == self.shape + (self.n_agents,), "invalid state shape"
            x = state
        else:
            x = np.array([self.np_random.uniform(low=low, high=high) for _ in range(self.n_agents)]).T

        self.state, self.signal, self.memory = self._split_state_vector(x)

        return self._get_obs()

    def _get_obs (self):
        '''
        Returns a function of the state corresponding to the
        observation available to the agent
        '''
        return np.vstack((self.state, self.signal, self.memory))

    def _get_state(self):
        '''
        Returns the **full** state of the system
        '''
        # self.state is (4, N)
        # self.signal is (1, N)
        # self.memory is (d_memory, N)
        # To get an array (d_tot, N) we vstack them
        return np.vstack((self.state, self.signal, self.memory))

    def _memory_step (self, rates):
        return self.memory + self.dt * f_memory(self.signal, self.memory, rates)

    def step(self, u):

        '''
        Computes one decision making step

        Input:
            u: action

        Output:
            obs: the observation after taking the action
            reward: the reward ensuing 'state_t, u, state_{t+1}'
            done: whether terminal state has been reached
            info: a dictionary with additional information
        '''

        info = {}

        info['state'] = self._get_state()
        info['signal'] = self.signal

        # clip the control within [0, max_u]
        # add a second axis to min_u and max_u, for the different agents
        u = np.clip(u, self.min_u[:,None], self.max_u[:,None])

        control = u[:self.d_control]
        rates = u[self.d_control:]

        # compute production and degradation rates
        # returns a pair of arrays, (d_state, n_agents)
        p, d = f_poni(self.state, control[:2])

        # deterministic step
        f = p - d
        x = self.state + self.dt * f
        s = self._signal_step(control)
        m = self._memory_step(rates)

        # in case there is noise, add it
        if self.noise is not None:
            # noise in the state
            g = np.random.normal(size=self.shape+(self.n_agents,))
            amp = np.sqrt(self.noise * self.dt * (p + d) )
            eta = amp * g[:self.d_state]
            x = x + eta
            # noise in the signal
            # amp = np.sqrt(self.noise * self.dt)
            # eta = amp * g[-self.d_signal:]
            # s = s + eta
        
        x = np.clip(np.vstack((x,s,m)), self.low[:,None], self.high[:,None])
        # keep state, signal and memory split (for convenience)
        self.state, self.signal, self.memory = self._split_state_vector(x)
        self.time += self.dt

        # distance of each cell state from the local target
        distance_sq = np.sum( (self.state - self.target)**2 , axis=0 )

        dones = distance_sq < self.radius**2
        info['dones'] = dones
        done = np.all(dones)

        '''
        calculate running costs/rewards
        - control
        - reward shaping (steps towards the target rewarded)
        '''
        # penalize control
        control_cost = np.sum( .5*self.eta*u**2, axis=0 ) * self.dt
        info["control_cost"] = control_cost

        # penalize state
        state_cost = self.cost(self.state, self.target, self.radius, 0.2) * self.dt
        info["state_cost"] = state_cost

        running_cost = control_cost + state_cost

        rewards = np.zeros(self.n_agents)
        if not done:
            rewards -= running_cost
        elif self.steps_beyond_done is None:
            # we do this when the target is first reached
            self.steps_beyond_done = 0
        else:
            # we do this if the agent keeps going, even if a terminal
            # state has been reached
            # what's the point? surely not during training, maybe
            # when testing the agent on continuing tasks
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            rewards = np.zeros(self.n_agents)

        return self._get_obs(), rewards, done, info



class PONINetwork_Diffusion_Memory_partial(PONINetwork_Diffusion_Memory):
    pass
