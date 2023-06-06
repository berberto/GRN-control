import gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from algorithms.AC import ActorCriticAgent
from algorithms.utils import plot_avg_scores
from datetime import datetime

ENV = "LunarLander-v2"
n_episodes = 3000


def set_filename(agent, env_name, n_episodes):
    dt_string = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    filename = f"plots/{env_name.split('-v')[0]}_actor-critic"
    dict_dims = {"fts":agent.features_dims,
                 "act":agent.actor_dims[:-1],
                 "val":agent.critic_dims[:-1]}
    for group, dims in dict_dims.items():
        if len(dims):
            filename += f"_{group}"
            for dim in dims:
                filename += f"-{dim}"
    filename += f"_lr-{agent.lr}_ep-{n_episodes}_{dt_string}"
    return filename


if __name__ == "__main__":
    env = gym.make(ENV)
    actions = env.action_space
    states = env.observation_space

    print(f"shape of state space = {states.shape}")
    print(f"number of actions = {actions.n}")

    agent = ActorCriticAgent(states.shape, actions.n,
            features_dims=[2048, 1536], actor_dims=None, critic_dims=None,
            lr=5e-6, gamma=0.99)

    # set filename
    filename = set_filename(agent, ENV, n_episodes)

    print(filename)

    scores = []
    for i in tqdm(range(n_episodes)):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)
            obs = obs_
            agent.learning_step(obs, reward, obs_, done)
            score += reward
            # if i%1000 == 0:
            #   env.render()
        scores.append(score)

    np.save(f"{filename}.npy", np.array(scores))
    plot_avg_scores(scores, f"{filename}.png")
