import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ActorNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, fc1_dims=400, fc2_dims=300, lr=1e-4,
                 checkpoint_dir="tmp/ddpg", filename="model", **kwargs):

        assert isinstance(input_dims, (tuple, list, np.ndarray)), "'input_dims' must be either list, tuple or numpy array"

        super(ActorNetwork, self).__init__(**kwargs)

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.lr = lr

        self.filename = filename
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.filename)
        
        # 2 hidden layers, batch-normalized
        self.fc1 = nn.Linear(*input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, n_actions)
        
        self.bn1 = nn.LayerNorm(self.fc1_dims) # nn.BatchNorm1d(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims) # nn.BatchNorm1d(self.fc2_dims)
        # why not normalizing before? like
        # self.bn1 = nn.LayerNorm(*input_dims) # nn.BatchNorm1d(*input_dims)
        # self.bn2 = nn.LayerNorm(self.fc1_dims) # nn.BatchNorm1d(self.fc1_dims)

        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        fout = 0.003
        self.mu.weight.data.uniform_(-fout, fout)
        self.mu.bias.data.uniform_(-fout, fout)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward (self, state):
        assert isinstance(state, T.Tensor), "'state' needs to be a torch.Tensor object"
        x = state
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
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
    def __init__(self, input_dims, n_actions, fc1_dims=400, fc2_dims=300, lr=1e-3, weight_decay=1e-2,
                 checkpoint_dir="tmp/ddpg", filename="model", **kwargs):
        assert isinstance(input_dims, (tuple, list, np.ndarray)), "'input_dims' must be either list, tuple or numpy array"
        
        super(CriticNetwork, self).__init__(**kwargs)

        self.lr = lr

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, filename)

        # 2 hidden layers, batch-normalized
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        
        self.bn1 = nn.LayerNorm(fc1_dims) # nn.BatchNorm1d(fc1_dims)
        self.bn2 = nn.LayerNorm(fc2_dims) # nn.BatchNorm1d(fc2_dims)
        # why not normalizing before? like
        # self.bn1 = nn.LayerNorm(*input_dims) # nn.BatchNorm1d(*input_dims)
        # self.bn2 = nn.LayerNorm(fc1_dims) # nn.BatchNorm1d(fc1_dims)

        # here we input the actions, together with the states
        self.fca = nn.Linear(n_actions, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        # initialization of the parameters of the layers is uniform [-1/sqrt(f), 1/sqrt(f)] by default
        # (we don't have to do anything to the parameters during initialization)

        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        fa = 1./np.sqrt(self.fca.weight.data.size()[0])
        self.fca.weight.data.uniform_(-fa, fa)
        self.fca.bias.data.uniform_(-fa, fa)        

        fout = 0.003
        self.q.weight.data.uniform_(-fout, fout)
        self.q.bias.data.uniform_(-fout, fout)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=weight_decay)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        assert isinstance(state, T.Tensor) and isinstance(action, T.Tensor), \
               "'state' and 'action' need to be torch.Tensor objects"
        # pre-process the state
        x = state
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)

        # input the actions after the second layer
        # first parse the action by a linear layer
        a = action
        a = self.fca(a)

        # then sum together with the partial result from the state-dependent
        # part of the calculation
        value = F.relu(T.add(x, a))
        value = self.q(value)

        return value

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
    print("pi = ", net)
    x = T.randn(6).view(-1,3)
    print("x = ", x)

    print(list(net.parameters()))

    exit()


    a = net.forward(x)
    print("a = ", a, "\n")

    print(" Test the CriticNetwork ")
    print("------------------------")
    net = CriticNetwork((3,), 2)
    print("Q = ", net)

    y = net.forward(x, a)
    
    print("v = ", y, "\n")