import gym
import numpy as np
from algorithms import TD3


ENV = "BipedalWalker-v3"
n_episodes = 1500
n_runs = 5
rendering = False
overwrite = False


if __name__ == "__main__":
    
    np.random.seed(0)
    env = gym.make(ENV)
    
    fc1_dims=400
    fc2_dims=300
    alpha=1e-3
    beta=1e-3
    gamma=0.99
    batch_size=100

    alg = TD3 (env, n_episodes, n_runs, fc1_dims=fc1_dims, fc2_dims=fc2_dims,
               alpha=alpha, beta=beta, gamma=gamma, batch_size=batch_size,
               overwrite=overwrite, rendering=rendering)

    alg.train()

    alg.test(10, rendering=rendering)

