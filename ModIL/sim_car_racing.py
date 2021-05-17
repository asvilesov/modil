import tensorflow as tf
import numpy as np 

'''Action Map'''
from skimage import color, transform
import itertools as it
action_map = np.array([k for k in it.product([-1, 0, 1], [1, 0], [0.2, 0])])

'''Initial Buffer'''
buff = np.random.rand(96,96,4)

def process_image(obs):
    return 2 * color.rgb2gray(obs) - 1.0

policy_l = tf.keras.models.load_model("./models/cr_policy_next2")

import gym
env_name = "CarRacing-v0"
env = gym.make(env_name)

'''Test Performance of Agent'''
observation = env.reset()
reward_total = 0
reward_history = []
steps = 0
for i in range(1000):
    env.render()
    # action = env.action_space.sample() # your agent here (this takes random actions)
    #print(observation.shape)

    observation = np.reshape(np.array([process_image(observation)]), newshape=(96,96,1))
    buff = np.concatenate((observation, buff[:,:,0:3]), axis=2)
    # print(observation)
    # observation = dynamics_data.transform_obs_forward(observation)
    # print(observation)
    # print()
    action_choices = policy_l(np.array([buff]))
    print(action_choices)
    action = np.argmax(action_choices)
    action = action_map[action]
    # if(i < 5):
    #     action[1] = 1
    #     action[2] = 0
    steps += 1
    print("Action taken: " + str(action))
    observation, reward, done, info = env.step(action)
    reward_total += reward

    if done:
        observation = env.reset()
        print("Reward Total: " + str(reward_total))
        print("Reset")
        print("steps: " + str(steps))
        reward_history.append(reward_total)
        reward_total=0
        steps = 0