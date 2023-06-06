import gym
import numpy as np
from .poni_signal import PONINetwork_Diffusion_Pattern
from .poni_memory import PONINetwork_Diffusion_Memory
from .sd import StochasticDiffusion


# PONI-pattern-v5
class PONINetwork_SD_Pattern (PONINetwork_Diffusion_Pattern):

    n_substeps = 5

    def __init__ (self, kappa=None, lam=None, size=None, d_memory=None):

        if kappa is None:
            kappa = 1.
        self.kappa = kappa
        print("self.kappa = ", self.kappa)

        if lam is None:
            lam = 0.15
        self.lam = lam
        print("self.lam = ", self.lam)

        if size is None:
            size = 500
        self.burst_size = int(size/50)
        self.size = size
        print("self.burst_size = ", self.burst_size)
        print("self.size = ", self.size)

        super(PONINetwork_SD_Pattern, self).__init__()

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

        self.sd = StochasticDiffusion(np.array([]), n_bins=self.n_agents, dt=self.dt/self.n_substeps,
                        kappa=self.kappa, lam=self.lam, burst_size=self.burst_size)

        self.state = x[:self.d_state, :]    # or x[:-self.d_signal, :]
        self.signal = x[-self.d_signal:, :] # or x[self.d_state:, :]
        return self._get_obs()

    def _signal_step (self, u):

        for _ in range(self.n_substeps):
            _signal, _ = self.sd.step()
        return _signal.reshape(1,-1)/self.sd.size


# PONI-pattern-v6
class PONINetwork_SD_Memory (PONINetwork_Diffusion_Memory):

    n_substeps = 5

    def __init__ (self, kappa=None, lam=None, size=None, d_memory=None):

        if kappa is None:
            kappa = 1.
        self.kappa = kappa
        print("self.kappa = ", self.kappa)

        if lam is None:
            lam = 0.15
        self.lam = lam
        print("self.lam = ", self.lam)

        if size is None:
            size = 500
        self.burst_size = int(size/50)
        self.size = size
        print("self.burst_size = ", self.burst_size)
        print("self.size = ", self.size)

        if d_memory is None:
            d_memory = 2
        self.d_memory = d_memory
        print("self.d_memory = ", self.d_memory)

        super(PONINetwork_SD_Memory, self).__init__(d_memory=self.d_memory)

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

        self.sd = StochasticDiffusion(np.array([]), n_bins=self.n_agents, dt=self.dt/self.n_substeps,
                        kappa=self.kappa, lam=self.lam, burst_size=self.burst_size)
        
        self.state, self.signal, self.memory = self._split_state_vector(x)

        return self._get_obs()

    def _signal_step (self, u):

        for _ in range(self.n_substeps):
            _signal, _ = self.sd.step()
        return _signal.reshape(1,-1)/self.sd.size


# PONI-pattern-v7
class PONINetwork_SD_Memory_Feedback (PONINetwork_Diffusion_Memory):
    n_substeps = 5

    def __init__ (self, kappa=None, lam=None, size=None, d_memory=None):

        if kappa is None:
            kappa = 1.
        self.kappa = kappa
        print("self.kappa = ", self.kappa)

        if lam is None:
            lam = 0.15
        self.lam = lam
        print("self.lam = ", self.lam)

        if size is None:
            size = 500
        self.burst_size = int(size/50)
        self.size = size
        print("self.burst_size = ", self.burst_size)
        print("self.size = ", self.size)

        if d_memory is None:
            d_memory = 2
        self.d_memory = d_memory
        print("self.d_memory = ", self.d_memory)

        self.tau_mem = 1.  # time scale for memory variables
        self.d_state = 4
        self.d_signal = 1
        self.d_control = 4
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
        self.max_u = np.ones(self.d_action, dtype=float)
        self.max_u[:2] = 2.
        self.max_u[-self.d_memory:] = 1.
        self.min_u = np.zeros(self.d_action, dtype=float)

        self.seed()

        self.max_time = 50.     # time horizon

        self.time = 0.          # timer

        self.action_space = gym.spaces.Box(
            low=np.zeros_like(self.max_u),
            high=self.max_u,
            shape=self.max_u.shape,
            dtype=np.float32
        )
        # 'high' is the array of upper bounds in each dimension
        # 'low' is that of the lower bounds
        # State includes agent state and signal received
        self.state_space = gym.spaces.Box(
            low=self.low,
            high=self.high,
            dtype=float
        )
        # if the environment has partial observability, 
        # define the observation space based on the 
        # dimensions that we want to observe
        try:
            self.observation_space = gym.spaces.Box(
                low=self.low[self.obs_dims],
                high=self.high[self.obs_dims],
                dtype=float
            )
        except:
            self.observation_space = self.state_space

        self.shaping_weight = None

        self.set_task_parameters()
        self.set_target_cost()

        self.reset()

    
    def reset (self, state=None):

        # print("self.d_action = ", self.d_action)
        # exit()
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

        self.sd = StochasticDiffusion(np.array([]), n_bins=self.n_agents, dt=self.dt/self.n_substeps,
                        kappa=self.kappa, lam=self.lam, burst_size=self.burst_size)
        
        self.state, self.signal, self.memory = self._split_state_vector(x)

        return self._get_obs()
        

    def _signal_step (self, u):

        for _ in range(self.n_substeps):
            _signal, _ = self.sd.step(prod=u[-2], degr=u[-1])
        return _signal.reshape(1,-1)/self.sd.size