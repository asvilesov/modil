# Dataset Creater/Manager for OpenAI Gym type Environments
# Less Efficient than buffer.py method, but easier to interface for lower dimension envs 
import numpy as np
import imblearn.over_sampling as imb
import os
from os.path import expanduser

#OS Helper function
home = expanduser("~")
def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

class dataset(object):

    def __init__(self, filename, max_obs = 100000, transform = True, oversampling = True, discrete_actions = True):

        #Init
        self.obs = self.actions = self.rewards = self.starts = None
        self.transform_obs = self.transform_actions = None
        self.num_observations = None

        self.X = self.Y = None
        self.X_size = self.Y_size = None
        self.action_map = None

        self.discrete_actions = True
        self.oversampling = True

        #Find filename
        self.file = find(filename, home)
        assert self.file != None
        
        #Extract npz file
        with np.load(self.file, allow_pickle=True) as data:
            lst = data.files
            for item in lst:
                print(item)
            #Cut dataset too max_obs size
            self.num_observations = len(data["obs"])
            if(self.num_observations > max_obs):
                self.num_observations = max_obs

            self.obs = data["obs"][0:self.num_observations]
            self.actions = data["actions"][0:self.num_observations]
            self.rewards = data["rewards"][0:self.num_observations]
            self.starts = data["episode_starts"][0:self.num_observations]

            self.obs_mean = np.zeros_like(self.obs[0])
            self.obs_std = np.ones_like(self.obs[0])
        
        #Discrete
        if(discrete_actions):
            self.max_actions = np.max(self.actions)
            self.min_actions = np.min(self.actions)
            self.action_map = np.arange(self.min_actions, self.max_actions + 1)
        
        #Processing dataset
        if(transform):
            self.transform()

        
    """----------------------------------------------------------- Active Dataset Testing  Functions ---------------------------------------------------------------"""
    #transforming to dynamics dataset
    def dynamics_init(self):
        self.X = []
        self.Y = []

        temp_action = []
        temp_xy_zip = None

        rewards = []
        reward = 0
        episodes = 0
        
        for i in range(self.num_observations-1):
            if not self.starts[i+1]:
                reward += self.rewards[i]
                self.X.append(np.concatenate( (self.obs[i],self.actions[i]), axis = None))
                self.Y.append(self.obs[i+1])
                temp_action.append(self.transform_actions_backward(self.actions[i]))
            else:
                rewards.append(reward)
                reward = 0
                episodes += 1
        
        print("rewards")
        print(np.mean(rewards))
        print(np.std(rewards))
        print(episodes)
        print(len(self.rewards))

        temp_xy_zip = np.array(list(zip(self.X, self.Y)), dtype=object)

        if(self.oversampling):
            temp_xy_zip, _ = self.oversample(temp_xy_zip, temp_action)
        
        np.random.shuffle(temp_xy_zip)
        self.X, self.Y = zip(*temp_xy_zip)
        
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)
        self.X_size = self.X.shape[1:]
        self.Y_size = self.Y.shape[1:]

    #transforming to mc dataset
    def mc_init(self):
        self.X = []
        self.Y = []

        temp_action = []
        temp_xy_zip = None
        
        for i in range(self.num_observations-1):
            if not self.starts[i+1]:
                self.X.append(self.obs[i])
                self.Y.append(self.obs[i+1])
                temp_action.append(self.transform_actions_backward(self.actions[i]))

        temp_xy_zip = np.array(list(zip(self.X, self.Y, temp_action)), dtype=object)
        
        if(self.oversampling):
            temp_xy_zip, _ = self.oversample(temp_xy_zip, temp_action)
            
        np.random.shuffle(temp_xy_zip)
        self.X, self.Y, _ = zip(*temp_xy_zip)

        self.X = np.array(self.X)
        self.Y = np.array(self.Y)
        self.X_size = self.X.shape[1:]
        self.Y_size = self.X.shape[1:]

    #transform current data
    def transform_obs_forward(self, data):
        return (data - self.obs_mean)/self.obs_std

    def transform_actions_forward(self, data):
        return (data - self.actions_mean)/self.actions_std

    def transform_obs_back(self, data):
        return (data * self.obs_std) + self.obs_mean

    def transform_actions_backward(self, data):
        action = (data * self.actions_std) + self.actions_mean
        if(self.discrete_actions):
            action = round(float(action))
        return action
        
    """------------------------------------------------------------ Dataset Initializer Helper Functions -------------------------------------------------------"""

    def oneHotEncoder_labels(self):
        '''
        @_labels - a list of _labels where each label is an integer from 0 to N-1, where N is the number of categories

        #creating one hot encoded _labels from a a numbered category
        '''
        n_categories = np.max(self._labels) + 1
        self._labels = np.eye(n_categories)[self._labels]

    def oversample(self, x, y):
        '''
        oversample minority actions in case of sparse actions
        '''
        ros = imb.RandomOverSampler(random_state=0)
        x, y = ros.fit_resample(x, y)
        self.check_balance()
        return x, y

    #Transforming to zero mean, unit std
    def transform(self):
        self.obs_mean = np.mean(self.obs, axis = 0)
        self.obs_std  = np.std(self.obs, axis = 0)
        self.obs = (self.obs - self.obs_mean)/self.obs_std
        
        #Could be Issues with Balancing of actions when training
        self.check_balance()
            
        ''' Normalize Actions '''
        # self.actions_mean = float(np.mean(self.actions, axis = 0))
        # self.actions_std  = float(np.std(self.actions, axis = 0))
        # self.actions = (self.actions - self.actions_mean)/self.actions_std
        ''' Normalize Actions Uniformly '''
        self.actions_mean = np.mean(self.action_map, axis = 0)
        self.actions_std  = np.std(self.action_map, axis = 0)
        self.actions = (self.actions - self.actions_mean)/self.actions_std
    
    def check_balance(self):
        """
        Check balance of targets
        returns: list where each element is percentage of samples for each label
        """
        self.action_info = np.zeros(int(self.max_actions+1))
        for i in self.actions:
            self.action_info[int(i)] += 1
        print(self.action_info)
        return self.action_info




if __name__ == "__main__":

    filename = "cartpole.npz"
    
    dynamics_data = dataset(filename, transform=True)
    dynamics_data.dynamics_init()

    mc_data = dataset(filename, transform=True)
    mc_data.mc_init()

    print(mc_data.actions, mc_data.actions_mean, mc_data.actions_std)


