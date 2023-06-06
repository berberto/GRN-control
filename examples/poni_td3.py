import os
import sys
import gym
import grn_control
import numpy as np
from algorithms import TD3

from poni_td3_plots import plot

ENV = sys.argv[1]
alpha = float(sys.argv[2])
beta = float(sys.argv[3])
gamma = float(sys.argv[4])
weight_decay = float(sys.argv[5])
seed = int(sys.argv[6])
# ENV = "PONI-(partial-)Nkx-v0" # Nkx target, deterministic, random init
# ENV = "PONI-(partial-)Nkx-v1" # Nkx target, stochastic,    random init
# ENV = "PONI-(partial-)Nkx-v2" # Nkx target, deterministic, localized init
# ENV = "PONI-(partial-)Nkx-v3" # Nkx target, stochastic,    localized init
# ENV = "PONI-(partial-)Oli-v0" # Oli target, deterministic, random init
# ENV = "PONI-(partial-)Oli-v1" # Oli target, stochastic,    random init
# ENV = "PONI-(partial-)Oli-v2" # Oli target, deterministic, localized init
# ENV = "PONI-(partial-)Oli-v3" # Oli target, stochastic,    localized init

## FULL OBSERVABILITY OF THE STATE
# ENV = "PONI-signal-Nkx-v0" # Nkx target, stochastic, localized init
# ENV = "PONI-signal-Oli-v0" # Nkx target, stochastic, localized init

## PARTIAL OBSERVABILITY OF THE STATE
# ENV = "PONI-signal-Nkx-v1" # Nkx target, stochastic, localized init
# ENV = "PONI-signal-Oli-v1" # Nkx target, stochastic, localized init


n_episodes = 2000
n_runs = 1
rendering = False
overwrite = False

vary_pars_training = False
vary_pars_testing = False

if __name__ == "__main__":
    
    np.random.seed(0)
    env = gym.make(ENV)
    
    fc1_dims=400
    fc2_dims=300
    batch_size=32

    # label to postfix to the environment name
    label = "vary_pars" if vary_pars_training else None

    alg = TD3 (env, n_episodes, n_runs,
               fc1_dims=fc1_dims, fc2_dims=fc2_dims,
               alpha=alpha, beta=beta, gamma=gamma,
               batch_size=batch_size, weight_decay=weight_decay,
               overwrite=overwrite, rendering=rendering,
               label=label, seed=seed)


    print("\nOutput directory:\n"\
          "-----------------\n"\
          f"{alg.checkpoint_dir}\n")

    alg.train(vary_parameters=vary_pars_training)

    # initialization rule for testing
    def init ():
        mu = np.array([1., 0., 0., 1.])
        while True:
            state = mu + 0.1 * np.random.normal(size=4)
            if np.all(state > 0):
                return state

    subdir = "_vary_pars" if vary_pars_testing else None

    test_episodes=1000
    state_dynamics, action_dynamics, scores, test_info = alg.test(test_episodes,
        init=init, rendering=rendering, overwrite=False,
        subdir=subdir, vary_parameters=vary_pars_testing)

    plot(alg, subdir=subdir)