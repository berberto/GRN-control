import os
import sys
import numpy as np
import gym
import grn_control
from algorithms import TD3_Multi
from poni_td3_pattern_plots import plot

ENV = sys.argv[1]
n_agents = int(sys.argv[2])
kappa = float(sys.argv[3])
size = int(sys.argv[4])
lam = float(sys.argv[5])
d_memory = int(sys.argv[6])
weight_decay = float(sys.argv[7])
seed = int(sys.argv[8])

# Pattern target, stochastic, localized init
## FULL OBSERVABILITY OF THE STATE
# ENV = "PONI-pattern-v0" # static signal
# ENV = "PONI-pattern-v2" # dynamic signal

## PARTIAL OBSERVABILITY OF THE STATE
# ENV = "PONI-pattern-v1" # static signal
# ENV = "PONI-pattern-v3" # dynamic signal
## FULL OBSERVABILITY, WITH MEMORY VARIABLES
# ENV = "PONI-pattern-v4" # dynamic signal

## STOCHASTIC SIGNAL (SPATIO-TEMPORAL CORRELATIONS)
# ENV = "PONI-pattern-v5" # NO MEMORY, fully observable
# ENV = "PONI-pattern-v6" # WITH MEMORY, fully observable


n_episodes = 50000
n_runs = 1
rendering = False
overwrite = False


if __name__ == "__main__":
    
    np.random.seed(0)

    # setup environment
    version = ENV.split("pattern-")[1]
    env_options = {}
    if version in ["v5", "v6", "v7"]:
        env_options["kappa"]=kappa
        env_options["size"]=size
        env_options["lam"]=lam
    if version in ["v4", "v6", "v7"]:
        env_options["d_memory"]=d_memory
    else:
        d_memory = 0

    env = gym.make(ENV, **env_options)
    env.set_target_cost(cost="lsq")

    fc1_dims=400
    fc2_dims=300
    alpha=1e-4
    beta=1e-5
    gamma=0.99

    batch_size=32


    alg = TD3_Multi (n_agents, env, n_episodes, n_runs,
               fc1_dims=fc1_dims, fc2_dims=fc2_dims,
               alpha=alpha, beta=beta, gamma=gamma,
               batch_size=batch_size, weight_decay=weight_decay,
               overwrite=overwrite, rendering=rendering, seed=seed)

    with open(os.path.join(f"{alg.checkpoint_dir}","parameters.txt"), "w") as f:
        f.write(f"coefficient cost for control = {alg.env.eta}\n")
        f.write(f"noise amplitude              = {alg.env.noise}\n")

    print("Checkpoints and outputs: \n\n", alg.checkpoint_dir, "\n")

    alg.train()

    # initialization rule for testing
    def init ():
        mu = np.array([1., 0., 0., 1., 0])
        if hasattr(env, "d_memory"):
            mu=np.concatenate((mu, np.zeros(alg.env.d_memory)))
        d = 0.1 * np.ones(len(mu))
        d[-1] = 0.001
        state = np.zeros((len(mu), n_agents))
        for n in range(n_agents):
            while True:
                state_ = mu + 0.1 * d * np.random.randn(len(mu))
                if np.all(state_ > 0):
                    state[:, n] = state_
                    break
        return state

    test_episodes=100

    # axes are:
    # 0 - episode; 1 - time; 2 - species; 3 - position/agent
    state_dynamics, action_dynamics, scores, test_info = alg.test(test_episodes,
            init=init, rendering=rendering, overwrite=False)

    plot(alg)
