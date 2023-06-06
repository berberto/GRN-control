import gym
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from algorithms import DDPGAgent
from algorithms.utils import plot_avg_scores


ENV = "LunarLanderContinuous-v2"
n_episodes = 1000
n_runs = 5

np.random.seed(0)

if __name__ == "__main__":
    env = gym.make(ENV)
    
    fc1_dims=400
    fc2_dims=300
    lr_actor=1e-4
    lr_critic=1e-3
    gamma=0.99
    batch_size=64

    actions = env.action_space
    states = env.observation_space
    min_score, max_score = env.reward_range

    print(f"shape of state space = {states.shape}")
    print(f"number of actions = {actions.shape}")

    file_id = f"{ENV.split('-v')[0]}_DDPG"
    file_id += f"_gamma:{gamma}"
    file_id += f"_fc1:{fc1_dims}_fc2:{fc2_dims}"
    file_id += f"_alpha:{lr_critic}_beta:{lr_actor}"
    file_id += f"_batch:{batch_size}_ep:{n_episodes}"
    
    checkpoint_dir = os.path.join("checkpoints", file_id)
    os.makedirs(checkpoint_dir)


    scores = np.zeros((n_runs,n_episodes))
    best_score = min_score
    for run in range(n_runs):
        # initialize the agent for every run
        agent = DDPGAgent(states.shape, actions.shape[0], fc1_dims=fc1_dims, fc2_dims=fc2_dims,
                lr_actor=lr_actor, lr_critic=lr_critic, gamma=gamma, checkpoint_dir=checkpoint_dir,
                batch_size=batch_size)
        
        print(f"Run {run} started")
        print("-----------------")
        print(agent)

        for i in range(n_episodes): # tqdm(range(n_episodes))
            obs = env.reset()
            agent.noise.reset()
            done = False
            score = 0
            while not done:
                env.render()
                action = agent.choose_action(obs)
                obs_, reward, done, info = env.step(action)
                agent.store(obs, action, reward, obs_, done)
                agent.learning_step()
                score += reward
                obs = obs_
            
            scores[run, i] = score
            avg = np.mean(scores[run,max(0,i-100):i+1])
            if avg > best_score:
                best_score = avg
                agent.save_models()
            print("episode %d,  score = %03.1f,   avg = %03.1f"%(i, score, avg))

        print(f"Run {run} finished")
        print("^^^^^^^^^^^^^^^^^^\n")

    filename = f"plots/{file_id}.png"
    plot_avg_scores(scores, filename, win=100)

