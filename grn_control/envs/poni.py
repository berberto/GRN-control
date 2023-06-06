import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from os import path
from .cost_functions import eps_relu_target
from .poni_aux import f_poni, default_pars

#######################################################################
#
#   Base class for PONI network environment
#
#######################################################################

class PONINetwork(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    noise = None # noise strength
    restart_low =  np.zeros(4)
    restart_high = np.ones(4)
    target = np.array([0., 0., 1., 0.])    # target to reach
    high = 1.2*np.ones(4, dtype=float)
    low = np.zeros_like(high, dtype=float)
    max_u = 2.*np.ones(2, dtype=float)
    min_u = np.zeros(2, dtype=float)

    dt = 0.05          # time-step
    eta = .1           # cost for control  
    viewer = None      # boh

    par_string = f"dt:{dt:.3f}_eta:{eta:.2f}"

    def __init__(self):
        self.seed()

        self.max_time = 50.     # time horizon

        self.radius = 0.2       # tolerance to target

        self.time = 0.          # timer

        self.action_space = spaces.Box(
            low=np.zeros_like(self.max_u),
            high=self.max_u,
            shape=self.max_u.shape,
            dtype=np.float32
        )
        # 'high' is the array of upper bounds in each dimension
        # 'low' is that of the lower bounds
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

        self.reset()

    def set_shaping_weight(self, weight):
        self.shaping_weight = weight

    @property
    def shape(self):
        return self.state_space.shape
    

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

        info['state'] = self.state.copy()

        # clip the control within [0, max_u]
        u = np.clip(u, self.min_u, self.max_u)

        # compute production and degradation rates
        p, d = f_poni(self.state, u, pars=self.pars)

        # deterministic step
        f = p - d
        x = self.state + self.dt * f

        # in case there is noise, add it
        if self.noise is not None:
            amp = np.sqrt(self.noise * self.dt * (p + d) )
            eta = amp * np.random.normal(size=self.shape)
            x = x + eta

        self.state = np.clip(x, self.low, self.high)
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
        state_cost = eps_relu_target(self.state, self.target, self.radius, 0.2) * self.dt
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

    def reset(self, state=None, vary_parameters=False, rel_var=0.05):
        '''
        Returns the OBSERVATION that is given in a random initial state
        '''
        self.pars = default_pars
        if vary_parameters:
            # vary each parameter by +- 5%, by default
            for key in self.pars.keys():
                self.pars[key] = 1. + 2.*rel_var*(np.random.rand() - 0.5)

        self.steps_beyond_done = None
        self.prev_shaping = None
        self.time = 0.
        low = self.restart_low
        high = self.restart_high
        if state is not None:
            assert state.shape == self.shape, "invalid state shape"
            self.state = state
        else:
            self.state = self.np_random.uniform(low=low, high=high)
        return self._get_obs()

    def _get_obs(self):
        '''
        Returns a function of the state corresponding to the
        observation available to the agent
        '''
        return self.state # it's a MDP

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


#######################################################################
#
#   Different target, initial conditions, noise
#
#######################################################################

class PONINetworkNoise(PONINetwork):
    noise = 5e-3 # noise strength

class PONINetworkHard(PONINetwork):
    noise = None # noise strength
    restart_low =  np.array([0.9, 0.0, 0.0, 0.9])
    restart_high = np.array([1.0, 0.1, 0.1, 1.0])

class PONINetworkHardNoise(PONINetwork):
    noise = 5e-3 # noise strength
    restart_low =  np.array([0.9, 0.0, 0.0, 0.9])
    restart_high = np.array([1.0, 0.1, 0.1, 1.0])


class PONINetwork_Olig(PONINetwork):
    noise = None # noise strength
    target = np.array([0., 1., 0., 0.])    # target to reach

class PONINetworkNoise_Olig(PONINetwork):
    noise = 5e-3 # noise strength
    target = np.array([0., 1., 0., 0.])    # target to reach

class PONINetworkHard_Olig(PONINetwork):
    noise = None # noise strength
    restart_low =  np.array([0.9, 0.0, 0.0, 0.9])
    restart_high = np.array([1.0, 0.1, 0.1, 1.0])
    target = np.array([0., 1., 0., 0.])    # target to reach

class PONINetworkHardNoise_Olig(PONINetwork):
    noise = 5e-3 # noise strength
    restart_low =  np.array([0.9, 0.0, 0.0, 0.9])
    restart_high = np.array([1.0, 0.1, 0.1, 1.0])
    target = np.array([0., 1., 0., 0.])    # target to reach


#######################################################################
#
#   With partial observability (only Olig and Nkx feedback)
#
#######################################################################

class PONINetwork_partial(PONINetwork):
    obs_dims = np.array([1,2]) # return Olig, Nkx only
    def _get_obs(self):
        return self.state[self.obs_dims]

class PONINetworkNoise_partial(PONINetworkNoise):
    obs_dims = np.array([1,2]) # return Olig, Nkx only
    def _get_obs(self):
        return self.state[self.obs_dims]

class PONINetworkHard_partial(PONINetworkHard):
    obs_dims = np.array([1,2]) # return Olig, Nkx only
    def _get_obs(self):
        return self.state[self.obs_dims]

class PONINetworkHardNoise_partial(PONINetworkHardNoise):
    obs_dims = np.array([1,2]) # return Olig, Nkx only
    def _get_obs(self):
        return self.state[self.obs_dims]

class PONINetwork_Olig_partial(PONINetwork_Olig):
    obs_dims = np.array([1,2]) # return Olig, Nkx only
    def _get_obs(self):
        return self.state[self.obs_dims]

class PONINetworkNoise_Olig_partial(PONINetworkNoise_Olig):
    obs_dims = np.array([1,2]) # return Olig, Nkx only
    def _get_obs(self):
        return self.state[self.obs_dims]

class PONINetworkHard_Olig_partial(PONINetworkHard_Olig):
    obs_dims = np.array([1,2]) # return Olig, Nkx only
    def _get_obs(self):
        return self.state[self.obs_dims]

class PONINetworkHardNoise_Olig_partial(PONINetworkHardNoise_Olig):
    obs_dims = np.array([1,2]) # return Olig, Nkx only
    def _get_obs(self):
        return self.state[self.obs_dims]

__all__ = [
        "PONINetwork",
        "PONINetworkNoise",
        "PONINetworkHard",
        "PONINetworkHardNoise",
        "PONINetwork_Olig",
        "PONINetworkNoise_Olig",
        "PONINetworkHard_Olig",
        "PONINetworkHardNoise_Olig",
        # 
        "PONINetwork_partial",
        "PONINetworkNoise_partial",
        "PONINetworkHard_partial",
        "PONINetworkHardNoise_partial",
        "PONINetwork_Olig_partial",
        "PONINetworkNoise_Olig_partial",
        "PONINetworkHard_Olig_partial",
        "PONINetworkHardNoise_Olig_partial"
]
