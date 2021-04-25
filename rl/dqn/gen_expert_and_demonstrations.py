import os
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
tf.disable_v2_behavior()
import gym
import _thread
import re
import sys
import pickle
import warnings

from rl.dqn.agent import CarRacingDQN

warnings.filterwarnings("ignore", category=UserWarning)

def save_checkpoint(path):
	if not os.path.exists(path):
		os.makedirs(path)
	p = os.path.join(path, "m.ckpt")
	saver.save(sess, p, dqn_agent.global_counter)
	print("saved to %s - %d" % (p, dqn_agent.global_counter))


def one_episode(path):
	
	reward, frames, demo = dqn_agent.play_episode(gen_demos=True)
	
	print("episode: %d, reward: %f, length: %d, total steps: %d" %
		  (dqn_agent.episode_counter, reward, frames, dqn_agent.global_counter))

	save_cond = (
		dqn_agent.episode_counter % save_freq_episodes == 0
		and path is not None
		and dqn_agent.do_training
	)
	if save_cond:
		save_checkpoint(path)


def input_thread(list):
	input("...enter to stop after current episode\n")
	list.append("OK")

def train_loop(path):
    """
    This just calls training function
    as long as we get input to stop
    """
    list = []
    _thread.start_new_thread(input_thread, (list,))
    while True:
        if list:
            break
        if dqn_agent.do_training and dqn_agent.episode_counter > train_episodes:
            break
        one_episode(path)

    print("done")


def demo_loop(save_path = ""):
	"""
	This just calls training function
	as long as we get input to stop
	"""
	demos =[]
	len_demos = 2
	list = []
	_thread.start_new_thread(input_thread, (list,))
	k = 0
	while k<len_demos:
		if list:
			break
		if dqn_agent.do_training and dqn_agent.episode_counter > train_episodes:
			break
		reward, frames, demo = dqn_agent.play_episode(gen_demos=True)
		demos.append(demo)
		print("episode: %d, reward: %f, length: %d, total steps: %d" %
			  (dqn_agent.episode_counter, reward, frames, dqn_agent.global_counter))

		k +=1 
	filename = save_path + 'demos.pkl'
	os.makedirs(os.path.dirname(filename), exist_ok=True)
	with open(filename, 'wb') as f:
		pickle.dump(demos, f, pickle.HIGHEST_PROTOCOL)

	print("done")


if __name__ == "__main__":

	#####################################################################################################
	# SETTINGS
	env_name = "CarRacing-v0"

	model_config = dict(
		min_epsilon=0.1,
		max_negative_rewards=12,
		num_frame_stack=3,
		frame_skip=3,
		train_freq=4,
		batchsize=64,
		epsilon_decay_steps=int(1e5),
		network_update_freq=int(1e3),
		min_experience_size=int(1e2),
		experience_capacity=int(4e4),
		gamma=0.95,
		soft_dqn=True
	)

	print(model_config)

	env = gym.make(env_name)

	# tf.reset_default_graph()
	dqn_agent = CarRacingDQN(env=env, **model_config)
	dqn_agent.build_graph()
	sess = tf.InteractiveSession()
	dqn_agent.session = sess

	# to start training or generating checkpoints:
	load_checkpoint = True
	checkpoint_path = "data/" + env_name + "/" + dqn_agent.algo_name + "/"
	train_episodes = 1050 #Do you need to train?
	save_freq_episodes = 200

	#####################################################################################################

	saver = tf.train.Saver(max_to_keep=100)

	if load_checkpoint:
		print("loading the latest checkpoint from %s" % checkpoint_path)
		ckpt = tf.train.get_checkpoint_state(checkpoint_path)
		assert ckpt, "checkpoint path %s not found" % checkpoint_path
		global_counter = int(re.findall("-(\d+)$", ckpt.model_checkpoint_path)[0])
		saver.restore(sess, ckpt.model_checkpoint_path)
		dqn_agent.global_counter = global_counter
	else:
		tf.global_variables_initializer().run()

	#####################################################################################################
	#Training

	if train_episodes > 0:
		print("now training... you can early stop with enter...")
		print("##########")
		sys.stdout.flush()
		train_loop(checkpoint_path)
		save_checkpoint()
		print("ok training done")


	#Demo Generation
	sys.stdout.flush()
	dqn_agent.max_neg_rewards = 100
	dqn_agent.do_training = False

	print("now just playing...")
	print("##########")
	sys.stdout.flush()
	demo_loop(checkpoint_path)

	print("That's it. Good bye")
	