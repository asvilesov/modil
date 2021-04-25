

from stable_baselines import DQN
from stable_baselines.gail import generate_expert_traj



model = DQN('MlpPolicy', 'CartPole-v1', verbose=1)
      # Train a DQN agent for 1e5 timesteps and generate 10 trajectories
      # data will be saved in a numpy archive named `expert_cartpole.npz`
num_episodes = 10000
generate_expert_traj(model, '../data/expert/cartpole' + str(num_episodes), n_timesteps=int(1e5), n_episodes=num_episodes)