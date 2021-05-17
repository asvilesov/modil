''' Markov Chain Estimation between continious Expert MC and agent induced MC'''

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

import data.datasets as dc
from policy import Policy, CnnPolicy
from model_dynamics import DynamicsModel, train_dynamicsModel, MarkovChainCNNModel, TransitionModel

import time #jupyter notebook bug - carriage returns



class mcEstimator(object):

    def __init__(self, config, mc_model, dynamics_model, policy, dataset):

        #Parameterized Functions
        self.mc_model = mc_model
        self.dynamics_model = dynamics_model
        self.policy = policy

        #Data
        self.dataset = dataset
        self.action_dim = config["action_size"]

        self.config = config

    def train_mc(self, x):
        epochs = self.config["epochs"]
        batch_size = self.config["batch_size"]

        for epoch in range(epochs):
            loss_history = []
            for batch in range(len(x)//batch_size):
                x_batch = x[batch*batch_size: min(len(x),(batch+1)*batch_size)]
                # y_batch = y[batch*batch_size: min(len(y),(batch+1)*batch_size)]
                loss = self.train_mc_batch(x_batch)
                loss_history.append(loss)
                print("Batch: " + str(batch) + "/" + str(len(x)//batch_size) + " - Loss: " + str(np.mean(loss)), end='\r')
            print("Epoch " + str(epoch) + " Training Loss: " + str(np.mean(loss_history)))    

    def train_mc_batch(self, x):
        """[summary]

        Args:
            x [state]: [NxF]
            
        Intermediate:
            mc_probs        [Expert Markob Chain Probabilities]:     [Nx1]
            action_probs    [Policy Action Probabilities]:           [NxA]
            transition_probs[Transition Probabilities]:              [NxFxA]
            agent_probs     [Agent Markov Chain Probabilities]:      [Nx1]

        Returns:
            loss [loss]
        """

        #x input is state => transform to p(s'|s) of agent
        mc_means = self.mc_model(x)
        #1. ones for scale because of scaling in dataset
        #2. Features are Indendent
        mc_distrib = tfp.distributions.Normal(loc=mc_means, scale=np.ones_like(mc_means)) 
        mc_probs = mc_distrib.prob(mc_means)

        #TODO not good way of generalizing actions #TODO Are action being fedin the right way
        trans_features = []
        for i in range(self.action_dim):
            # print(self.dataset.transform_actions_forward(i))
            # print(i)
            # print()
            x_with_action = np.append(x, np.array([self.dataset.transform_actions_forward(i)]*x.shape[0])[np.newaxis].T, axis=1)
            #_with_action = np.append(x, np.array([i-0.5]*x.shape[0])[np.newaxis].T, axis=1)
            
            # print(x.shape)
            # print(x_with_action.shape)
            predict = self.dynamics_model(x_with_action)
            # print(predict[0:4])
            trans_features.append(predict)
        transition_means = np.array(trans_features)
        transition_means = np.transpose(transition_means, axes=[1,2,0])
        # print(transition_means.shape)
        # transition_means = self.dynamics_model(trans_probs)

        with tf.GradientTape() as tape:
            # print(x)
            action_probs = self.policy(x)
            # print("Action Probs ", action_probs[0:4])
            # print("Transition Probs ", transition_means[0:4])
            agent_means = tf.matmul(transition_means, tf.expand_dims(action_probs, 2))
            # print(agent_means.shape)
            agent_means = tf.reduce_sum(agent_means, axis = 2)
            # agent_means = tf.transpose(agent_means, perm=[1,0])
            # print("Expert Means: ", mc_means.shape,mc_means[0:4])
            # print("Agent Means: ", agent_means.shape,agent_means[0:4])

            agent_probs = mc_distrib.prob(agent_means)

            # print("mc_probs", mc_probs[0:4])
            # print("agnet_probs", agent_probs[0:4])

            loss = self.config["loss"](mc_probs, agent_probs) + self.policy.regularization_loss()

        grads = tape.gradient(loss, self.policy.trainable_weights)
        self.config["optimizer"].apply_gradients(zip(grads, self.policy.trainable_weights))

        return loss

class imageMcEstimator(object):

    def __init__(self, config, mc_model, dynamics_model, policy, dataset):

        #Parameterized Functions
        self.mc_model = mc_model
        self.dynamics_model = dynamics_model
        self.policy = policy

        #Data
        self.dataset = dataset
        self.action_dim = config["action_size"]

        self.config = config

    def train_mc(self):
        epochs = self.config["epochs"]
        self.batch_size = self.config["batch_size"]

        num_batches = self.dataset.counter // self.batch_size

        for epoch in range(epochs):
            
            loss_history = []
            for batch in range(num_batches):
                
                x_batch = self.dataset.sample_mini_batch(self.batch_size)
                prev_states = x_batch['prev_state']
                loss = self.train_mc_batch(prev_states)
                loss_history.append(loss)
                print("\rBatch: " + str(batch) + "/" + str(num_batches) + " - Loss: " + str(np.mean(loss)), end=' ')
                time.sleep(0)
            print("Epoch " + str(epoch) + " Training Loss: " + str(np.mean(loss_history)))
            print()    

    def train_mc_batch(self, x):
        """[summary]

        Args:
            x [state]: [NxF]
            
        Intermediate:
            mc_probs        [Expert Markob Chain Probabilities]:     [Nx1]
            action_probs    [Policy Action Probabilities]:           [NxA]
            transition_probs[Transition Probabilities]:              [NxFxA]
            agent_probs     [Agent Markov Chain Probabilities]:      [Nx1]

        Returns:
            loss [loss]
        """

        #x input is state => transform to p(s'|s) of agent
        mc_means = self.mc_model(x)
        #1. ones for scale because of scaling in dataset
        #2. Features are Indendent
        mc_distrib = tfp.distributions.Normal(loc=mc_means, scale=np.ones_like(mc_means)) 
        mc_probs = mc_distrib.prob(mc_means)

        #TODO not good way of generalizing actions #TODO Are action being fed in the right way?
        trans_features = []
        for i in range(self.action_dim):
            # print(self.dataset.transform_actions_forward(i))
            # print(i)
            # print()
            
            x_actions = np.eye(self.action_dim)[[i]*self.batch_size, :]
            #_with_action = np.append(x, np.array([i-0.5]*x.shape[0])[np.newaxis].T, axis=1)
            
            # print(x.shape)
            # print(x_with_action.shape)
            predict = self.dynamics_model(x, x_actions)

            trans_features.append(predict)

        transition_means = np.array(trans_features)
        # print(transition_means.shape)
        # print(transition_means.shape[1:])
        # print(transition_means.shape[1:] + (0,))
        transition_means = np.transpose(transition_means, axes=(1,2,3,4,0))
        # print(transition_means.shape)
        # transition_means = self.dynamics_model(trans_probs)

        with tf.GradientTape() as tape:
            # print(x)
            action_probs = self.policy(x)
            # print("Action Probs ", action_probs[0:4])
            # print("Transition Probs ", transition_means[0:4])
            agent_means = tf.multiply(transition_means, tf.reshape(action_probs, (action_probs.shape[0],1,1,1,action_probs.shape[1])))
            # print(agent_means.shape)
            agent_means = tf.reduce_sum(agent_means, axis = 4)
            # agent_means = tf.transpose(agent_means, perm=[1,0])
            # print("Expert Means: ", mc_means.shape,mc_means[0:4])
            # print("Agent Means: ", agent_means.shape,agent_means[0:4])

            agent_probs = mc_distrib.prob(agent_means)

            # print("mc_probs", mc_probs[0:4])
            # print("agnet_probs", agent_probs[0:4])

            loss = self.config["loss"](mc_probs, agent_probs) #+ self.policy.regularization_loss()

        grads = tape.gradient(loss, self.policy.trainable_weights)
        self.config["optimizer"].apply_gradients(zip(grads, self.policy.trainable_weights))

        return loss



if __name__ == "__main__":

    import gym

    config = {  
            "loss": tf.keras.losses.KLD, 
            "optimizer": tf.keras.optimizers.Adam(learning_rate=0.00001),
            "validation_split": 0.10,
            "max_obs": int(1e5),
            "epochs": 16,
            "batch_size": 64,
            "state_size": 2, 
            "action_size": 3 #2
             }

    from data.experience_history import ExperienceHistory

    data_buff = ExperienceHistory(num_frame_stack=4, capacity=int(1e6), pic_size=(96,96))
    expert_data_path = "data/expert/demos.pkl"
    data_buff.loadExpertHistory(expert_data_path)
    test_sample = data_buff.sample_mini_batch(64)
    print(test_sample['prev_state'].shape)
    print(test_sample['next_state'].shape)
    print(test_sample['actions'].shape)
    print(test_sample['actions'])

    mc = MarkovChainCNNModel((96,96, 4))

    sample_recons = mc(test_sample['prev_state'])

    print(sample_recons.shape)

    trans = TransitionModel((96,96,4), 12)
    sample_recons = trans(test_sample['prev_state'], np.random.rand(64, 12))

    print(sample_recons.shape)






