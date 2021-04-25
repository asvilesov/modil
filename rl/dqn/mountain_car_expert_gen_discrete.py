
from stable_baselines import DQN
from stable_baselines.gail import generate_expert_traj



model = DQN('MlpPolicy', 'MountainCar-v0', verbose=1)
      # Train a DQN agent for 1e5 timesteps and generate 10 trajectories
      # data will be saved in a numpy archive named `expert_cartpole.npz`
num_episodes = 1000
generate_expert_traj(model, '../data/expert/MountainCarDiscrete' + str(num_episodes), n_timesteps=int(2e5), n_episodes=num_episodes)