import gym
import os
import pickle
import numpy as np
import torch as T
import matplotlib.pyplot as plt
from .agents import TD3Agent, TD3MultiAgent
from ..utils import plot_avg_scores
from ..processors import ActionProcessor, ActionProcessor_Multi


class TD3 (object):
    def __init__(self, env, n_episodes, n_runs,
                 fc1_dims=400, fc2_dims=300,
                 alpha=1e-3, beta=1e-3, gamma=0.99,
                 weight_decay=0.0,
                 batch_size=100, overwrite=False,
                 verbose=True, rendering=False,
                 label=None, seed=0,
                 ):
        self.env = env
        self.env_name = self.env.unwrapped.spec.id

        self.n_episodes = n_episodes
        self.n_runs = n_runs
        self.rendering = rendering
        self.scores = np.nan * np.ones((self.n_runs,self.n_episodes))

        '''
        All environments have an observation_space attribute.
        Environments that yield partial observations
        also have a state_space attribute.
        We use the information from this state_space
        when testing a model, to get the full dynamics.
        Normally, gym environments are MDPs, and only have
        observation_space defined, in which case input_dims
        and state_dims will correspond.
        '''
        self.observations = self.env.observation_space
        self.input_dims = self.observations.shape
        if hasattr(self.env, "state_space"):
            self.partial_obs = True
            self.states = self.env.state_space
        else:
            self.partial_obs = False
            self.states = self.observations
        self.state_dims = self.states.shape
        
        self.actions = self.env.action_space
        self.n_actions = self.actions.shape[0]

        self.min_score, self.max_score = self.env.reward_range

        self.action_bounds = self.actions.low, self.actions.high
        self.processor = ActionProcessor(*self.action_bounds)

        self.file_id = f"{self.env_name}_TD3" 
        self.file_id += '_'+label if label != None else ''
        self.file_id = os.path.join(self.file_id, f"fc1:{fc1_dims}_fc2:{fc2_dims}")
        self.file_id = os.path.join(self.file_id, f"alpha:{alpha}_beta:{beta}_wd:{weight_decay:.1e}")
        self.file_id = os.path.join(self.file_id, f"batch:{batch_size}_ep:{n_episodes}")
        self.file_id = os.path.join(self.file_id, f"gamma:{gamma:.3f}")
        try:
            self.file_id += f"_mem:{self.env.d_memory}"
        except:
            pass
        # 
        try:
            self.file_id += f"_{self.env.par_string}"
        except:
            pass
        self.file_id = os.path.join(self.file_id, f"{seed:d}")
        
        self.checkpoint_dir = os.path.join("checkpoints", self.file_id)

        self.options = {
            "fc1_dims": fc1_dims,
            "fc2_dims": fc2_dims,
            "lr_critic": alpha,
            "lr_actor": beta,
            "weight_decay": weight_decay,
            "gamma": gamma,
            "batch_size": batch_size,
            "checkpoint_dir": self.checkpoint_dir,
            "buffer_size":100000,
            "seed": seed,
        }

        self._set_agent()

        self.verbose = verbose
        if verbose:
            print(f"shape of observ space = {self.input_dims}")
            print(f"shape of state space = {self.states.shape}")
            print(f"number of actions = {self.actions.shape}")
            print(f"bound of actions = {self.action_bounds}")
            print(self.agent)

        self.trained = False
        try:
            os.makedirs(self.checkpoint_dir, exist_ok=overwrite)
        except FileExistsError:
            print("Existing checkpoint -- agent assumed to be trained")
            self.trained = True


    def test_actor(self, state_array):
        assert isinstance(state_array, np.ndarray), "input must be a numpy array"
        state = T.tensor(state_array, dtype=T.float).view(-1, *self.input_dims)
        state = state.to(self.agent.actor.device)
        with T.no_grad():
            action = self.agent.actor.forward(state).cpu().detach().numpy()
        return self.processor.transform(action)

    def test_critic(self, state_array, action_array):
        assert isinstance(state_array, np.ndarray) and isinstance(action_array, np.ndarray), \
               "inputs must be numpy arrays"
        state = T.tensor(state_array, dtype=T.float).view(-1, *self.input_dims)
        action = T.tensor(action_array, dtype=T.float).view(-1, self.n_actions)
        assert state.shape[0] == action.shape[0], \
               "batch dimensions of 'state' and 'action' must match"

        state = state.to(self.agent.critic.device)
        action = action.to(self.agent.critic.device)
        with T.no_grad():
            value = self.agent.critic.forward(state,action).cpu().detach().numpy()

        return value

    def _set_agent (self):
        self.agent = TD3Agent(self.input_dims, self.n_actions, **self.options)

    def _choose_action (self, obs):
        return self.agent.choose_action(obs)

    def _store (self, obs, action, reward, obs_, done, info=None):
        self.agent.store(obs, action, reward, obs_, done)

    def train (self, **reset_kwargs):

        if self.trained:
            print("Skipping training")
            return

        best_score = self.min_score
        for run in range(self.n_runs):

            # initialize the agent from scratch
            self._set_agent()
            
            # save agent (and scores) at the beginning of every run
            # (end of the previous run)
            self.save_scores()
            
            if self.verbose:
                print(f"Run {run} started")
                print("-----------------")

            for i in range(self.n_episodes): # tqdm(range(n_episodes))
                obs = self.env.reset(**reset_kwargs)
                done = False
                score = 0
                discount = 1.
                while not done:
                    if self.rendering:
                        self.env.render()
                    action = self._choose_action(obs)
                    act_ = self.processor.transform(action)
                    obs_, reward, done, info = self.env.step(act_)
                    self._store(obs, action, reward, obs_, done, info=info)
                    self.agent.learning_step()
                    score += reward * discount
                    discount *= self.agent.gamma
                    obs = obs_
                

                self.scores[run, i] = np.sum(score)
                avg = np.mean(self.scores[run,max(i-100,0):i+1])
                if avg > best_score:
                    best_score = avg
                    self.agent.save_models()
                if self.verbose:
                    print("episode %d,  score = %03.1f,   avg = %03.1f"%(i, np.sum(score), avg))

                if i % 10 == 0:
                    # save and plot scores every some episodes
                    self.save_scores()
                    self.plot_training()

            self.env.close()

            if self.verbose:
                print(f"Run {run} finished")
                print("^^^^^^^^^^^^^^^^^^\n")

    def load_scores (self):
        filename = os.path.join(self.checkpoint_dir, "training.npy")
        self.scores = np.load(filename)

    def save_scores (self):
        filename = os.path.join(self.checkpoint_dir, "training.npy")
        np.save(filename, self.scores)

    def plot_training (self):
        filename = os.path.join(self.checkpoint_dir, "training.svg")
        plot_avg_scores(self.scores, filename, win=100)

    def _setup_dynamics (self, n_episodes):
        '''
        Prepare arrays to store dynamics, with correct shapes
        '''
        state_dynamics = np.zeros((n_episodes, self.env._max_episode_steps) + self.state_dims)
        action_dynamics = np.zeros((n_episodes, self.env._max_episode_steps, self.n_actions))
        return state_dynamics, action_dynamics

    def __test (self, n_episodes, init=None, rendering=True,
            extra_info=True, **reset_kwargs):
        state_dynamics, action_dynamics = self._setup_dynamics(n_episodes)

        scores = np.zeros(n_episodes)

        # list of dictionaries, one per episode
        test_info = [] if extra_info else None
        for i in range(n_episodes): # tqdm(range(n_episodes))
            test_info_ = {}

            print(i)
            if init is not None:
                state = init()
                try:
                    obs = self.env.reset(state=state, **reset_kwargs)
                except TypeError:
                    obs = self.env.reset(**reset_kwargs)
            else:
                obs = self.env.reset(**reset_kwargs)


            done = False
            score = 0
            step_counter = 0
            discount = 1.
            while not done:
                # print(step_counter)
                if rendering:
                    self.env.render()
                action = self._choose_action(obs)
                act_ = self.processor.transform(action)
                obs_, reward, done, info = self.env.step(act_)
                action_dynamics[i, step_counter] = act_
                # save extra info
                if extra_info:
                    for key, val in info.items():
                        if step_counter == 0:
                            test_info_[key] = []
                        try:
                            test_info_[key].append(val)
                        except KeyError:
                            continue
                '''
                The state we save in `state_dynamics` is either the observation
                if the task is a MDP --so `partial_obs` is False-- or the full
                state of the environment otherwise. In this latter case, the 
                full state is provided with the `info` dictionary
                '''
                if self.partial_obs:
                    try:
                        state_dynamics[i, step_counter] = info['state']
                    except KeyError:
                        raise KeyError("There is no 'state' returned in 'info'")
                else:
                    state_dynamics[i, step_counter] = obs
                
                score += reward * discount
                discount *= self.agent.gamma
                obs = obs_
                step_counter += 1
            
            test_info.append(test_info_)
            scores[i] = np.sum(score)
            print("episode %d,  score = %03.1f"%(i, np.sum(score)))

        self.env.close()

        return state_dynamics, action_dynamics, scores, test_info

    def test (self, n_episodes, init=None, rendering=True,
        overwrite=True, extra_info=True, subdir=None, **reset_kwargs):
        '''
        Testing the model on a number of episodes

        init: callable, or None
            if callable, it should take no arguments and return an array valid
            as state of the environment: this is called to generate the initial
            state at the beginning of every episode.
            If None (default) the default rule defined by the `reset` method
            in the environment is applied.
            This allows to change the initialization of the state from training
            to testing.
        '''

        self.agent.load_models()

        savedir = self.checkpoint_dir
        if subdir != None:
            savedir = os.path.join(self.checkpoint_dir, subdir)
            os.makedirs(savedir, exist_ok=True)
        
        try:
            if overwrite:
                raise Exception
            print("Trying to load saved test dynamics...")
            state_dynamics = np.load(os.path.join(savedir, "state_dynamics.npy"))
            action_dynamics = np.load(os.path.join(savedir, "action_dynamics.npy"))
            scores = np.load(os.path.join(savedir, "test_scores.npy"))
            test_info = pickle.load( open(os.path.join(savedir, "test_info.pkl"), "rb") )

        except:
            print("Testing anew...")
            state_dynamics, action_dynamics, scores, test_info = self.__test(n_episodes,
                init=init, rendering=rendering, extra_info=extra_info, **reset_kwargs)
            np.save(os.path.join(savedir, "state_dynamics.npy"), state_dynamics)
            np.save(os.path.join(savedir, "action_dynamics.npy"), action_dynamics)
            np.save(os.path.join(savedir, "test_scores.npy"), scores)
            pickle.dump( test_info, open(os.path.join(savedir, "test_info.pkl"), "wb") )

        return state_dynamics, action_dynamics, scores, test_info



class TD3_Multi (TD3):
    def __init__(self, n_agents, env, n_episodes, n_runs,
                 fc1_dims=400, fc2_dims=300,
                 alpha=1e-3, beta=1e-3, gamma=0.99,
                 weight_decay=0.0,
                 batch_size=100, overwrite=False,
                 verbose=True, rendering=False,
                 seed=0,
                ):
        self.n_agents = n_agents
        self.env = env
        self.env.set_n_agents(self.n_agents)
        self.env_name = self.env.unwrapped.spec.id

        self.n_episodes = n_episodes
        self.n_runs = n_runs
        self.rendering = rendering
        self.scores = np.nan * np.ones((self.n_runs,self.n_episodes))

        '''
        All environments have an observation_space attribute.
        Environments that yield partial observations
        also have a state_space attribute.
        We use the information from this state_space
        when testing a model, to get the full dynamics.
        Normally, gym environments are MDPs, and only have
        observation_space defined, in which case input_dims
        and state_dims will correspond.
        '''
        self.observations = self.env.observation_space
        self.input_dims = self.observations.shape
        if hasattr(self.env, "state_space"):
            self.partial_obs = True
            self.states = self.env.state_space
        else:
            self.partial_obs = False
            self.states = self.observations
        self.state_dims = self.states.shape
        
        self.actions = self.env.action_space
        self.n_actions = self.actions.shape[0]

        self.min_score, self.max_score = self.env.reward_range

        self.action_bounds = self.actions.low, self.actions.high
        self.processor = ActionProcessor_Multi(*self.action_bounds)

        self.file_id = f"{self.env_name}_TD3Multi_n:{n_agents}"
        self.file_id = os.path.join(self.file_id, f"fc1:{fc1_dims}_fc2:{fc2_dims}")
        self.file_id = os.path.join(self.file_id, f"alpha:{alpha}_beta:{beta}_wd:{weight_decay:.1e}")
        self.file_id = os.path.join(self.file_id, f"batch:{batch_size}_ep:{n_episodes}")
        self.file_id = os.path.join(self.file_id, f"gamma:{gamma:.3f}")
        try:
            self.file_id += f"_mem:{self.env.d_memory}"
        except:
            pass
        # 
        try:
            self.file_id += f"_{self.env.par_string}"
        except:
            pass
        
        try:
            subdir =  f"kappa:{self.env.kappa:.2f}"
            subdir += f"_lam:{self.env.lam:.2f}"
            subdir += f"_size:{self.env.size:05d}"
            self.file_id = os.path.join(self.file_id, subdir)

        except:
            pass
        self.file_id = os.path.join(self.file_id, f"{seed:d}")
        self.checkpoint_dir = os.path.join("checkpoints", self.file_id)

        self.options = {
            "fc1_dims": fc1_dims,
            "fc2_dims": fc2_dims,
            "lr_critic": alpha,
            "lr_actor": beta,
            "weight_decay": weight_decay,
            "gamma": gamma,
            "batch_size": batch_size,
            "checkpoint_dir": self.checkpoint_dir,
            "buffer_size":100000,
            "seed": seed,
        }

        self._set_agent()

        self.verbose = verbose
        if verbose:
            print(f"number of agents = {self.n_agents}")
            print(f"shape of observ space = {self.input_dims}")
            print(f"shape of state space = {self.states.shape}")
            print(f"number of actions = {self.actions.shape}")
            print(f"bound of actions = {self.action_bounds}")
            print(self.agent)

        self.trained = False
        try:
            os.makedirs(self.checkpoint_dir, exist_ok=overwrite)
        except FileExistsError:
            print("Existing checkpoint -- agent assumed to be trained")
            self.trained = True


    def _set_agent (self):
        self.agent = TD3MultiAgent(self.n_agents, self.input_dims, self.n_actions, **self.options)

    def _choose_action(self, obs):
        '''
        obs has shape (dim, n_agents) -- best way to pass to the dynamical system
        Need to transpose and use agent axis as batch axis for forward method in 
        ActorNetwork
        '''
        return self.agent.choose_action(obs.T).T

    def _store (self, obs, action, reward, obs_, done, info=None):
        '''
        First axis of arguments must be the agent index, but as returned
        by the environment, this would be the last.
        Transposing everything that is not a scalar.
        `done` is taken from the `info` dictionary, which contains
        additional information about the environment (see `gym` doc)
        '''
        try:
            dones = info['dones']
        except:
            print("something wrong with \"done\" in \"_store\"")
            dones = (done * np.ones(self.n_agents)).astype(bool)

        self.agent.store(obs.T, action.T, reward, obs_.T, dones)

    def save_scores (self):
        filename = os.path.join(self.checkpoint_dir, "training.npy")
        np.save(filename, self.scores/self.n_agents)

    def plot_training (self):
        filename = os.path.join(self.checkpoint_dir, "training.svg")
        plot_avg_scores(self.scores/self.n_agents, filename, win=100)

    def _setup_dynamics (self, n_episodes):
        '''
        Prepare arrays to store dynamics, with correct shapes
        '''
        state_dynamics = np.zeros((n_episodes, self.env._max_episode_steps, *self.state_dims, self.n_agents))
        action_dynamics = np.zeros((n_episodes, self.env._max_episode_steps, self.n_actions, self.n_agents))
        return state_dynamics, action_dynamics


class TD3_reshaping (TD3):
    '''
    Class for training a grn control which includes reward shaping.
    It subtract the reward shaping term from the return when calculating
    the score over an episode.

    '''
    def __init__(self, *td3_args, shaping_discount=0.995, **td3_kwargs):

        self.shaping_discount = shaping_discount
        super(TD3_reshaping, self).__init__(*td3_args, **td3_kwargs)


    def train (self):

        if self.trained:
            print("Skipping training")
            return

        best_score = self.min_score
        for run in range(self.n_runs):
            # initialize the agent for every run
            self.agent = TD3Agent(self.input_dims, self.n_actions, **self.options)
            
            if self.verbose:
                print(f"Run {run} started")
                print("-----------------")

            self.env.set_shaping_weight(1.)
            for i in range(self.n_episodes):
                obs = self.env.reset()
                done = False
                score = 0
                shaping = 0
                while not done:
                    if self.rendering:
                        self.env.render()
                    action = self._choose_action(obs)
                    obs_, reward, done, info = self.env.step(self.processor.transform(action))
                    self._store(obs, action, reward, obs_, done)
                    self.agent.learning_step()
                    score += reward
                    if 'shaping' in info:
                        shaping += info['shaping']
                    obs = obs_
                
                self.env.set_shaping_weight(self.env.shaping_weight * self.shaping_discount)
                
                self.scores[run, i] = score - shaping
                avg = np.mean(scores[run,max(i-100,0):i+1])
                if avg > best_score:
                    best_score = avg
                    self.agent.save_models()
                if self.verbose:
                    print("episode %d,  score = %03.1f  of which shaping = %03.1f,   avg net score = %03.1f"%(i, score, shaping, avg))

            self.env.close()

            self.plot_training()

            if self.verbose:
                print(f"Run {run} finished")
                print("^^^^^^^^^^^^^^^^^^\n")