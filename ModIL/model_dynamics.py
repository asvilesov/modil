# Model Environment Dynamics (State Transitions) 

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class DynamicsModel(tf.keras.Model):
    '''Dynamics Model can be used for both Transition and MC function in low dim action and state spaces '''    
    def __init__(self, input_dim, output_dim, activation_func = tf.nn.relu, h_neurons=256, regularizer=tf.keras.regularizers.l2(0.0001)):
        super(DynamicsModel, self).__init__()

        #vector input
        if(len(input_dim) == 1):
            self.l1 = tf.keras.layers.Dense(units=h_neurons, activation=activation_func, kernel_regularizer=regularizer)
            self.l2 = tf.keras.layers.Dense(units=h_neurons, activation=activation_func, kernel_regularizer=regularizer)
            self.l3 = tf.keras.layers.Dense(units=output_dim[0], kernel_regularizer=regularizer)
        #image input
        else:
            raise NotImplementedError

    def call(self, input, training=False):
        x = self.l1(input)
        x = self.l2(x)
        x = self.l3(x)

        return x


def train_dynamicsModel(model, dataset, epochs=100):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss = tf.losses.mean_squared_error, metrics =["mae", "acc"])
    model.fit(dataset.X, dataset.Y, batch_size = 64, epochs = epochs, validation_split = 0.10, shuffle = True)

class MarkovChainCNNModel(tf.keras.Model):
    def __init__(self, input_dim, activation_func = tf.nn.relu, regularizer=tf.keras.regularizers.l2(0.0001)):
        super(MarkovChainCNNModel, self).__init__()

        self.c1 = tf.keras.layers.Conv2D(16, 3, activation=activation_func, kernel_regularizer=regularizer, padding="same")
        self.c2 = tf.keras.layers.Conv2D(32, 3, activation=activation_func, kernel_regularizer=regularizer, padding="same")
        self.c3 = tf.keras.layers.Conv2D(64, 3, activation=activation_func, kernel_regularizer=regularizer, padding="same")

        self.maxpool = tf.keras.layers.MaxPool2D()
        self.upsamp = tf.keras.layers.UpSampling2D()

        self.c1_d = tf.keras.layers.Conv2D(64, 3, activation=activation_func, kernel_regularizer=regularizer, padding="same")
        self.c2_d = tf.keras.layers.Conv2D(32, 3, activation=activation_func, kernel_regularizer=regularizer, padding="same")
        self.c3_d = tf.keras.layers.Conv2D(input_dim[2], 3, activation=activation_func, kernel_regularizer=regularizer, padding="same")

    def call(self, input, training=False):

        x1 = self.c1(input)
        x2 = self.maxpool(x1)
        x2 = self.c2(x2)
        x3 = self.maxpool(x2)
        x3 = self.c3(x3)

        x = self.upsamp(x3)
        x = tf.concat((x2, x), axis = 3)
        x = self.c1_d(x)
        
        x = self.upsamp(x)
        x = tf.concat((x1, x), axis = 3)
        x = self.c2_d(x)

        x = self.c3_d(x)

        return x

def trainMarkovChainCNN(model, experience_history, config):
    batch_size = config["batch_size"]
    optimizer = config["optimizer"]
    loss_func = config["loss"]
    epochs = config["epochs"]
    batches = experience_history.counter // batch_size

    for i in range(epochs):
        for j in range(batches):
            '''Get batch'''
            batch = experience_history.sample_mini_batch(batch_size)
            current_state = batch['prev_state']
            target_state  = batch['next_state']
            '''predicts'''
            with tf.GradientTape() as tape:
                predict_next_state = model(current_state)
                loss = loss_func(predict_next_state, target_state)
            '''apply gradients'''
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            '''log'''
            print("Batch: " + str(j) + "/" + str(batches) + " - Loss: " + str(np.mean(loss)), end='\n')

class TransitionModel(tf.keras.Model):
    def __init__(self, input_dim, action_dim, activation_func = tf.nn.relu, regularizer=tf.keras.regularizers.l2(0.0001)):
        super(TransitionModel, self).__init__()

        self.input_dim = input_dim
        self.action_dim = action_dim

        self.c1 = tf.keras.layers.Conv2D(16, 3, activation=activation_func, kernel_regularizer=regularizer, padding="same")
        self.c2 = tf.keras.layers.Conv2D(32, 3, activation=activation_func, kernel_regularizer=regularizer, padding="same")
        self.c3 = tf.keras.layers.Conv2D(64, 3, activation=activation_func, kernel_regularizer=regularizer, padding="same")
        self.c4 = tf.keras.layers.Conv2D(1, 3, activation=activation_func, kernel_regularizer=regularizer, padding="same")

        self.maxpool = tf.keras.layers.MaxPool2D()

        self.f4 = tf.keras.layers.Dense(input_dim[0]/8*input_dim[1]/8, activation=activation_func, kernel_regularizer=regularizer)

        self.upsamp = tf.keras.layers.UpSampling2D()

        self.c1_d = tf.keras.layers.Conv2D(32 + 64, 3, activation=activation_func, kernel_regularizer=regularizer, padding="same")
        self.c2_d = tf.keras.layers.Conv2D(32 + 64, 3, activation=activation_func, kernel_regularizer=regularizer, padding="same")
        self.c3_d = tf.keras.layers.Conv2D(16 + 32, 3, activation=activation_func, kernel_regularizer=regularizer, padding="same")
        self.c4_d = tf.keras.layers.Conv2D(input_dim[2], 3, activation=activation_func, kernel_regularizer=regularizer, padding="same")

    def call(self, input_state, input_action, training=False):

        x1 = self.c1(input_state)
        x2 = self.maxpool(x1)
        x2 = self.c2(x2)
        x3 = self.maxpool(x2)
        x3 = self.c3(x3)
        x4 = self.maxpool(x3)
        x4 = self.c4(x4)

        x4_flat = tf.keras.layers.Flatten()(x4)
        # print(x4_flat.shape)
        # print(input_action.shape)
        x4_flat = tf.concat((x4_flat, input_action), axis = 1)
        x4_flat = self.f4(x4_flat)
        x4 = tf.reshape(x4_flat, shape=(-1, int(self.input_dim[0]/8), int(self.input_dim[1]/8), 1))
        # print(x4.shape)

        x = self.upsamp(x4)
        x = tf.concat((x3, x), axis = 3)
        x = self.c1_d(x)

        x = self.upsamp(x)
        x = tf.concat((x2, x), axis = 3)
        x = self.c2_d(x)
        
        x = self.upsamp(x)
        x = tf.concat((x1, x), axis = 3)
        x = self.c3_d(x)

        x = self.c4_d(x)

        return x

def trainTransitions(model, experience_history, config):
    batch_size = config["batch_size"]
    optimizer = config["optimizer"]
    loss_func = config["loss"]
    epochs = config["epochs"]
    batches = experience_history.counter // batch_size

    for i in range(epochs):
        for j in range(batches):
            '''Get batch'''
            batch = experience_history.sample_mini_batch(batch_size)
            current_state = batch['prev_state']
            #convert actions to one hot
            actions  = batch['actions']
            actions  = np.eye(model.action_dim)[actions, :]
            target_state  = batch['next_state']
            '''predicts'''
            with tf.GradientTape() as tape:
                predict_next_state = model(current_state, actions)
                loss = loss_func(predict_next_state, target_state)
            '''apply gradients'''
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            '''log'''
            print("Batch: " + str(j) + "/" + str(batches) + " - Loss: " + str(np.mean(loss)), end='\n')




if __name__ == "__main__":

    import data.datasets as dc 

    config = {  
            "loss": tf.keras.losses.KLD, 
            "optimizer": tf.keras.optimizers.Adam(learning_rate=0.0001),
            "validation_split": 0.10,
            "max_obs": int(1e5),
            "epochs": 10,
            "batch_size": 64,
            "state_size": 4,
            "action_size": 2
             }


    #Import dataset
    dynamics_data = dc.dataset("MountainCarDiscrete1000.npz", max_obs=config["max_obs"])
    
    #Model environment dynamics s' <- (s,a)
    dynamics_data.dynamics_init()
    print(dynamics_data.X)
    cartpole_model = DynamicsModel(dynamics_data.X_size, dynamics_data.Y_size)
    train_dynamicsModel(cartpole_model, dynamics_data, epochs=config["epochs"])
    
    #Model Markov Chain dynamics
    dynamics_data.mc_init()
    cartpole_expert_mc = DynamicsModel(dynamics_data.X_size, dynamics_data.Y_size)
    train_dynamicsModel(cartpole_expert_mc, dynamics_data, epochs=config["epochs"])

    print("Number of observations trained on: " + str(dynamics_data.num_observations))
    


    