from matplotlib import pyplot as plt
import numpy as np

'''All mean reward values are a result of 10 experiments'''

'''CartPole'''
dataset_size = np.array([1e1,    1e2,    1e3,    1e4,   1e5])

expert = np.ones_like(dataset_size) * 500

mean_reward_modil  = np.array([16.954, 80.5,   232,    428.223, 461])
std_reward_modil   = np.array([14.35,  88.89,  105,    91.3639, 58])

plt.plot(dataset_size, expert, 'r-')
plt.plot(dataset_size, mean_reward_modil, 'b-')
plt.fill_between(dataset_size, np.clip(mean_reward_modil-std_reward_modil, 0, 500), np.clip(mean_reward_modil+std_reward_modil, 0, 500), alpha=0.5)

plt.legend(["Expert", "ModIL"])
plt.xscale("log")
plt.title("Model Based Imitation Learning: Cartpole-v0")
plt.xlabel("Expert Observations - (500 obs/episode)")
plt.ylabel("Reward")

plt.show()

'''Mountain Car'''
dataset_size = np.array([1e1,    2e2,    1e3,    1e4,   1e5])
expert = np.ones_like(dataset_size) * -125
mean_reward_modil  = np.array([-200, -183.58,   -158.75,    -143.17, -136.22])
std_reward_modil   = np.array([0,  29.15,  31.43,    9.752, 12.89])

plt.plot(dataset_size, expert, 'r-')
plt.plot(dataset_size, mean_reward_modil, 'b-')
plt.fill_between(dataset_size, np.clip(mean_reward_modil-std_reward_modil, -200, 0), np.clip(mean_reward_modil+std_reward_modil, -200, 0), alpha=0.5)

plt.legend(["Expert", "ModIL"])
plt.xscale("log")
plt.title("Model Based Imitation Learning: Mountaincar-v0")
plt.xlabel("Expert Observations - (200 obs/episode)")
plt.ylabel("Reward")

plt.show()