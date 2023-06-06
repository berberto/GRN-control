import gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from algorithms import REINFORCEAgent
from datetime import datetime
dt_string = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

ENV = "LunarLander-v2"

if __name__ == "__main__":
	env = gym.make(ENV)
	n_episodes = 3000
	actions = env.action_space
	states = env.observation_space
	scores = []
	avg_scores = []

	agent = REINFORCEAgent(states.shape, lr=0.0005)
	for i in tqdm(range(n_episodes)):
		obs = env.reset()
		done = False
		score = 0
		while not done:
			action = agent.choose_action(obs)
			obs_, reward, done, info = env.step(action)
			score += reward
			agent.reward_memory.append(reward)
			obs = obs_
			# if i%1000 == 0:
			# 	env.render()
		agent.update_policy()
		scores.append(score)
		avg_scores.append(np.mean(scores[-100 if i > 100 else 0 : ]))
	# env.close()
	plt.plot(avg_scores)
	plt.savefig(f"lunarlander_REINFORCE-{dt_string}.png")
	plt.show()
