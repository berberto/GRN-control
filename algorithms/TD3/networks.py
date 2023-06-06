import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ActorNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, fc1_dims=400, fc2_dims=300, lr=1e-3, weight_decay=0.0,
                 checkpoint_dir="tmp/td3", filename="model_actor"):

        assert isinstance(input_dims, (tuple, list, np.ndarray)), "'input_dims' must be either list, tuple or numpy array"

        super(ActorNetwork, self).__init__()

        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.lr = lr

        self.filename = filename
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.filename)
        
        # 2 hidden layers, batch-normalized
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=weight_decay)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward (self, state):
        assert isinstance(state, T.Tensor), "'state' needs to be a torch.Tensor object"
        x = state
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = T.tanh(self.mu(x))

        return x


    def save_checkpoint(self, verbose=False):
        if verbose:
            print("saving checkpoint")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, verbose=False):
        if verbose:
            print("loading checkpoint")
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))



class CriticNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, fc1_dims=400, fc2_dims=300, lr=1e-3, weight_decay=0.0,
                 checkpoint_dir="tmp/td3", filename="model_critic"):
        assert isinstance(input_dims, (tuple, list, np.ndarray)), "'input_dims' must be either list, tuple or numpy array"
        
        super(CriticNetwork, self).__init__()

        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.lr = lr

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, filename)

        # (it may become a mess with high-rank state/action tensors!!)
        self.fc1 = nn.Linear(self.input_dims[0] + self.n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=weight_decay)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        assert isinstance(state, T.Tensor) and isinstance(action, T.Tensor), \
               "'state' and 'action' need to be torch.Tensor objects"
        # concatenate state and action in one unique tensor
        # (it may become a mess with high-rank state/action tensors!!)
        x = T.cat((state, action), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.q(x)

        return x

    def save_checkpoint(self, verbose=False):
        if verbose:
            print("saving checkpoint")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, verbose=False):
        if verbose:
            print("loading checkpoint")
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))




if __name__ == "__main__":
    print(" Test the ActorNetwork ")
    print("------------------------")
    net = ActorNetwork((3,), 2, fc1_dims=10, fc2_dims=5)
    # print("pi = ", net)
    x = T.randn(6 * 5 * 4).view(4,5,2,3)
    print(x.shape)
    print("x = \n", x, "\n")

    a = net.forward(x)
    print(a.shape)
    print("a = \n", a, "\n")

    print(" Test the CriticNetwork ")
    print("------------------------")
    net = CriticNetwork((3,), 2)
    # print("Q = ", net)

    c = T.cat((x,a),dim=-1)
    print(c.shape)
    print("cat(a,b) = \n", c, "\n")

    y = net.forward(x, a)
    
    print(y.shape)
    print("v = \n", y, "\n")