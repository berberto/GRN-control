import numpy as np

class ReplayMemory():
    def __init__(self, input_dims, n_actions, buffer_size=1000000):
        assert isinstance(input_dims, tuple), "'input_dims' must be tuple (shape of state space)"
        assert isinstance(n_actions, int), "'n_actions' must be int (dimension of action space or number of actions)"
        self.buffer_size = buffer_size

        self.state_buffer = np.zeros((buffer_size, *input_dims))
        self.action_buffer = np.zeros((buffer_size, n_actions))
        self.newstate_buffer = np.zeros((buffer_size, *input_dims))
        self.reward_buffer = np.zeros(buffer_size)
        self.done_buffer = np.zeros(buffer_size, dtype=bool)

        self.reset()

    def store (self, state, action, reward, state_, done):
        '''
        Stores the elements sequence (s_t, a_t, r_{t+1}, s_{t+1}, done)
        into separate arrays.

        '''
        self.mem_index = self.mem_counter % self.buffer_size

        self.state_buffer[self.mem_index] = state
        self.action_buffer[self.mem_index] = action
        self.reward_buffer[self.mem_index] = reward
        self.newstate_buffer[self.mem_index] = state_
        self.done_buffer[self.mem_index] = done

        self.mem_counter += 1

    def reset (self):
        self.mem_counter = 0
        self._mem_size = 0

    @property
    def mem_size(self):
        self._mem_size = min(self.mem_counter, self.buffer_size)
        return self._mem_size
    
    def batch(self, batch_size):
        '''
        Returns a tuple of numpy arrays

        Arrays have length defined by 'batch_size'

        '''
        # self.mem_size = min(self.mem_counter, self.buffer_size)
        ids = np.random.choice(self.mem_size, batch_size,
                              replace=False)

        states = self.state_buffer[ids]
        actions = self.action_buffer[ids]
        rewards = self.reward_buffer[ids]
        states_ = self.newstate_buffer[ids]
        dones = self.done_buffer[ids]

        return states, actions, rewards, states_, dones

    def __len__(self):
        return self.mem_size

    def __repr__(self):
        info = ''
        info += f'mem_counter = {self.mem_counter}, mem_size = {self.mem_size}\n\n'
        info += f'state_buffer = \n {self.state_buffer[:self.mem_size]}, {self.state_buffer.shape}\n\n'
        info += f'action_buffer = \n {self.action_buffer[:self.mem_size]}, {self.action_buffer.shape}\n\n'
        info += f'newstate_buffer = \n {self.newstate_buffer[:self.mem_size]}, {self.newstate_buffer.shape}\n\n'
        info += f'reward_buffer = \n {self.reward_buffer[:self.mem_size]}, {self.reward_buffer.shape}\n\n'
        info += f'done_buffer = \n {self.done_buffer[:self.mem_size]}, {self.done_buffer.shape}'
        return info


class ReplayMemory_MultiAgent (ReplayMemory):
    def __init__ (self, n_agents, input_dims, n_actions, buffer_size=10000):
        assert isinstance(input_dims, tuple), "'input_dims' must be tuple (shape of state space)"
        assert isinstance(n_actions, int), "'n_actions' must be int (dimension of action space or number of actions)"
        self.buffer_size = buffer_size
        self.n_agents = n_agents

        self.state_buffer = np.zeros((self.buffer_size, self.n_agents, *input_dims))
        self.action_buffer = np.zeros((self.buffer_size, self.n_agents, n_actions))
        self.newstate_buffer = np.zeros((self.buffer_size, self.n_agents, *input_dims))
        self.reward_buffer = np.zeros((self.buffer_size, self.n_agents))
        self.done_buffer = np.zeros((self.buffer_size, self.n_agents), dtype=bool)

        self.reset()



if __name__ == "__main__":

    n_agents = 5
    input_dims = (2,3)
    n_actions = 2
    buffer_size = 4

    buffer = ReplayMemory_MultiAgent(n_agents, input_dims, n_actions, buffer_size=buffer_size)
    # buffer = ReplayMemory(input_dims, n_actions, buffer_size=buffer_size)

    def sample ():
        state = np.random.randn(n_agents, *input_dims).squeeze()
        action = np.random.randn(n_agents, n_actions).squeeze()
        reward = np.random.randn(n_agents).squeeze()
        state_ = np.random.randn(n_agents, *input_dims).squeeze()
        done = np.random.randint(2, size=n_agents).astype(bool).squeeze()
        return (
                state,
                action,
                reward,
                state_,
                done,
               )

    for _ in range(10):
        buffer.store(*sample())

    print(buffer)
    exit()
