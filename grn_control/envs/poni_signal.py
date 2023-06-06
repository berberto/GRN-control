import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from os import path
from .cost_functions import eps_relu_target, eps_relu_target_pattern
from .cost_functions import quadratic_target, quadratic_target_pattern
from .poni_aux import f_poni, f_signal, poni_target, poni_colors_array
    

#######################################################################
#
#   Base class for PONI network environment
#
#######################################################################

class PONINetwork_Signal(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }
    d_state = 4
    d_signal = 1
    d_action = 2
    noise = 5e-3 # None # noise strength
    restart_low =  np.array([0.9, 0.0, 0.0, 0.9, 0.0])
    restart_high = np.array([1.0, 0.1, 0.1, 1.0, 0.1])
    target = np.array([0., 0., 1., 0.])    # target to reach
    high = 1.2*np.ones(d_state + d_signal, dtype=float)
    high[-d_signal:] = 10. # very high bound for Shh
    low = np.zeros_like(high, dtype=float)
    max_u = 2.*np.ones(d_action, dtype=float)
    min_u = np.zeros(d_action, dtype=float)

    dt = 0.05          # time-step
    viewer = None      # boh

    def __init__(self):
        self.seed()

        self.max_time = 50.     # time horizon

        self.time = 0.          # timer

        self.action_space = spaces.Box(
            low=np.zeros_like(self.max_u),
            high=self.max_u,
            shape=self.max_u.shape,
            dtype=np.float32
        )
        # 'high' is the array of upper bounds in each dimension
        # 'low' is that of the lower bounds
        # State includes agent state and signal received
        self.state_space = spaces.Box(
            low=self.low,
            high=self.high,
            dtype=float
        )
        # if the environment has partial observability, 
        # define the observation space based on the 
        # dimensions that we want to observe
        try:
            self.observation_space = spaces.Box(
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

    def set_task_parameters (self, radius=0.2, eta=0.05):
        self.radius = radius # tolerance to target
        self.eta = eta       # cost for control

    def set_target_cost (self, cost='lsq'):
        self.cost_ = cost
        if cost == 'erelu':
            self.cost = eps_relu_target
        elif cost == 'lsq':
            self.cost = quadratic_target
        else:
            raise ValueError("invalid cost option")

    def set_shaping_weight(self, weight):
        self.shaping_weight = weight

    @property
    def shape(self):
        return self.state_space.shape

    @property
    def par_string (self):
        ps = ""
        ps += f"eta:{self.eta}"
        ps += f"_q:{self.cost_}"
        # ps += f"_rad:{self.radius}"
        return ps

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

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

        info['state'] = np.append(self.state, self.signal)
        info['signal'] = self.signal

        # clip the control within [0, max_u]
        u = np.clip(u, self.min_u, self.max_u)

        # compute production and degradation rates
        p, d = f_poni(self.state, u)

        # deterministic step
        f = p - d
        x = self.state + self.dt * f
        s = self.signal + self.dt * f_signal(self.time, 0., self.signal)

        # in case there is noise, add it
        if self.noise is not None:
            # noise in the state
            g = np.random.normal(size=self.shape)
            amp = np.sqrt(self.noise * self.dt * (p + d) )
            eta = amp * g[:-1]
            x = x + eta
            # noise in the signal
            amp = np.sqrt(self.noise * self.dt)
            eta = amp * g[-1]
            s = s + eta

        x = np.clip(np.append(x,s), self.low, self.high)
        # keep state and signal split (for convenience)
        self.state = x[:self.d_state]    # or x[:-self.d_signal]
        self.signal = x[-self.d_signal:] # or x[self.d_state:]
        self.time += self.dt

        distance_sq = np.sum( (self.state - self.target)**2 )
        # print(self.time, self.state, distance_sq, u)

        # done = bool( distance_sq < self.radius**2 )
        done = False

        '''
        calculate running costs/rewards
        - control
        - reward shaping (steps towards the target rewarded)
        '''
        # penalize control
        control_cost = np.sum( .5*self.eta*u**2 ) * self.dt
        info["control_cost"] = control_cost

        # penalize state
        state_cost = self.cost(self.state, self.target, self.radius, 0.2) * self.dt
        info["state_cost"] = state_cost

        running_cost = control_cost + state_cost
        
        # reward steps towards the target
        if self.shaping_weight is not None:
            shaping = - np.sqrt(distance_sq)/self.dt
            if self.prev_shaping is not None:
                reward_shaping = (shaping - self.prev_shaping) * self.shaping_weight
                info["shaping"] = reward_shaping
                # subtract from the running cost the reward from reward-shaping
                running_cost -= reward_shaping
            self.prev_shaping = shaping

        reward = 0
        if not done:
            reward -= running_cost
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
            reward = 0.

        return self._get_obs(), reward, done, info

    def reset(self, state=None):
        '''
        Returns the OBSERVATION that is given in a random initial state
        '''
        self.steps_beyond_done = None
        self.prev_shaping = None
        self.time = 0.
        low = self.restart_low
        high = self.restart_high
        if state is not None:
            assert state.shape == self.shape, "invalid state shape"
            x = state

        else:
            x = self.np_random.uniform(low=low, high=high)
        self.state = x[:self.d_state]    # or x[:-self.d_signal]
        self.signal = x[-self.d_signal:] # or x[self.d_state:]
        return self._get_obs()

    def _get_obs(self):
        '''
        Returns a function of the state corresponding to the
        observation available to the agent
        '''
        return np.append(self.state, self.signal)

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-.1, 1.1, -.1, 1.1) # left, right, bottom, top
            # create a circle of given radius
            # and set the color
            point = rendering.make_circle(radius=.025, res=30)
            point.set_color(0, 0, 0)
            self.point_transform = rendering.Transform()
            point.add_attr(self.point_transform)
            self.viewer.add_geom(point)

        # move the circle around based on the state
        oli, nkx = self.state[1], self.state[2]
        self.point_transform.set_translation(oli, nkx)     

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class PONINetwork_Signal_Nkx (PONINetwork_Signal):
    target = np.array([0., 0., 1., 0.])    # target to reach

class PONINetwork_Signal_Oli (PONINetwork_Signal):
    target = np.array([0., 1., 0., 0.])    # target to reach

class PONINetwork_Signal_Nkx_partial (PONINetwork_Signal):
    target = np.array([0., 0., 1., 0.])    # target to reach
    obs_dims = np.array([1,2,-1]) # return Olig, Nkx and Shh only
    def _get_obs(self):
        return np.hstack((self.state, self.signal))[self.obs_dims]

class PONINetwork_Signal_Oli_partial (PONINetwork_Signal):
    target = np.array([0., 1., 0., 0.])    # target to reach
    obs_dims = np.array([1,2,-1]) # return Olig, Nkx and Shh only
    def _get_obs(self):
        return np.hstack((self.state, self.signal))[self.obs_dims]


class PONINetwork_Signal_Pattern (PONINetwork_Signal):
    n_agents = 10
    positions = np.linspace(0.,1.,n_agents)
    target = poni_target(positions)

    def set_target_cost (self, cost='lsq'):
        self.cost_ = cost
        if cost == 'erelu':
            self.cost = eps_relu_target_pattern
        elif cost == 'lsq':
            self.cost = quadratic_target_pattern
        else:
            raise ValueError("invalid cost option")

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

        self.state = x[:self.d_state, :]    # or x[:-self.d_signal, :]
        self.signal = x[-self.d_signal:, :] # or x[self.d_state:, :]
        return self._get_obs()

    def set_n_agents (self, n):
        self.n_agents = n
        self.positions = np.linspace(0.,1.,self.n_agents)
        self.target = poni_target(self.positions)
        self.reset()

    def _get_obs (self):
        '''
        Returns a function of the state corresponding to the
        observation available to the agent
        '''
        # self.state is (4, N)
        # self.signal is (1, N)
        # To get an array (5, N) we vstack them
        return np.vstack((self.state, self.signal))

    def _signal_step (self, u):
        '''
        Update the signal based on action `u`
        '''
        return np.exp( - self.positions / 0.15 ) # static gradient

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

        info['state'] = np.vstack((self.state, self.signal))
        info['signal'] = self.signal

        # clip the control within [0, max_u]
        # add a second axis to min_u and max_u, for the different agents
        u = np.clip(u, self.min_u[:,None], self.max_u[:,None])

        # compute production and degradation rates
        # returns a pair of arrays, (d_state, n_agents)
        p, d = f_poni(self.state, u)

        # deterministic step
        f = p - d
        x = self.state + self.dt * f
        # s = self.signal + self.dt * f_signal(self.time, self.positions, self.signal)
        s = self._signal_step(u)

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
        
        x = np.clip(np.vstack((x,s)), self.low[:,None], self.high[:,None])
        # keep state and signal split (for convenience)
        self.state = x[:self.d_state]    # or x[:-self.d_signal]
        self.signal = x[-self.d_signal:] # or x[self.d_state:]
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

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-.1, 1.1, -.1, 1.1) # left, right, bottom, top
            # create a circle of given radius
            # and set the color
            points = [rendering.make_circle(radius=.01, res=10) for _ in range(self.n_agents)]
            self.point_transforms = [rendering.Transform() for _ in range(self.n_agents)]

            # set alpha channel as soft-max
            # alpha channel for each gene (N_cells, genes)
            alpha = np.exp(self.target.T/.3)
            alpha /= np.sum(alpha, axis=1)[:,None]

            colors = np.matmul(alpha, poni_colors_array)
            for point, transform, color in zip(points, self.point_transforms, colors):
                point.set_color(*color)
                point.add_attr(transform)
                self.viewer.add_geom(point)

        # move the circle around based on the state
        oli, nkx = self.state[1], self.state[2]
        for n, transform in enumerate(self.point_transforms):
            transform.set_translation(oli[n], nkx[n])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


class PONINetwork_Signal_Pattern_partial (PONINetwork_Signal_Pattern):
    obs_dims = np.array([1,2,-1]) # return Olig, Nkx and Shh only
    def _get_obs (self):
        '''
        Returns a function of the state corresponding to the
        observation available to the agent
        '''
        return np.vstack((self.state, self.signal))[self.obs_dims]


class PONINetwork_Diffusion_Pattern (PONINetwork_Signal_Pattern):
    kappa = 0.1
    def set_timescale (self, tau=10.):
        self.kappa = 1./tau

    def _signal_step (self, u):
        '''
        Update the signal based on action `u`
        '''
        return self.signal + self.dt * f_signal (self.time, self.positions, self.signal, kappa=self.kappa)

class PONINetwork_Diffusion_Pattern_partial (PONINetwork_Diffusion_Pattern):
    def _signal_step (self, u):
        '''
        Update the signal based on action `u`
        '''
        return self.signal + self.dt * f_signal (self.time, self.positions, self.signal, kappa=self.kappa)


__all__ = [
        "PONINetwork_Signal_Nkx",
        "PONINetwork_Signal_Oli",
        "PONINetwork_Signal_Nkx_partial",
        "PONINetwork_Signal_Oli_partial",
        "PONINetwork_Signal_Pattern",
        "PONINetwork_Signal_Pattern_partial",
        "PONINetwork_Diffusion_Pattern",
        "PONINetwork_Diffusion_Pattern_partial",
]
