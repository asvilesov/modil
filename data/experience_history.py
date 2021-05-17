import numpy as np
import pickle


class ExperienceHistory:
    """
    This saves the agent's experience in windowed cache.
    Each frame is saved only once but state is stack of num_frame_stack frames

    In the beginning of an episode the frame-stack is padded
    with the beginning frame
    """

    def __init__(self,
            num_frame_stack=4,
            capacity=int(1e5),
            pic_size=(96, 96)
    ):
        self.num_frame_stack = num_frame_stack
        self.capacity = capacity
        self.pic_size = pic_size
        self.counter = 0
        self.frame_window = None
        self.init_caches()
        self.expecting_new_episode = True
        #sample wo replacement
        self.random_idx = np.array([])
        self.batch_num = 0
    
    def loadExpertHistory(self, file_name):

        startEpisode = True
        with open(file_name, 'rb') as f:
            demos = pickle.load(f)
        
        for demo in demos:
            for obs, act, done, rew in demo:
                if startEpisode:
                    frame_idx = self.counter % self.max_frame_cache
                    self.frame_window = np.repeat(frame_idx, self.num_frame_stack)
                    self.frames[frame_idx] = obs
                    self.expecting_new_episode = False
                    startEpisode = False
                else:
                    self.add_experience(obs, act, done, rew)
                if done:
                    startEpisode = True
        print("Expert Data loaded with ", self.counter, " frames.", self.counter)

    def add_experience(self, frame, action, done, reward):
        assert self.frame_window is not None, "start episode first"
        self.counter += 1
        frame_idx = self.counter % self.max_frame_cache
        exp_idx = (self.counter - 1) % self.capacity

        self.prev_states[exp_idx] = self.frame_window
        self.frame_window = np.append(self.frame_window[1:], frame_idx)
        self.next_states[exp_idx] = self.frame_window
        self.actions[exp_idx] = action
        self.is_done[exp_idx] = done
        self.frames[frame_idx] = frame
        self.rewards[exp_idx] = reward
        if done:
            self.expecting_new_episode = True

    def start_new_episode(self, frame):
        # it should be okay not to increment counter here
        # because episode ending frames are not used
        assert self.expecting_new_episode, "previous episode didn't end yet"
        frame_idx = self.counter % self.max_frame_cache
        self.frame_window = np.repeat(frame_idx, self.num_frame_stack)
        self.frames[frame_idx] = frame
        self.expecting_new_episode = False

    def sample_mini_batch(self, n):
        count = min(self.capacity, self.counter)
        batchidx = np.random.randint(count, size=n)

        prev_frames = np.transpose(self.frames[self.prev_states[batchidx]], axes=(0,2,3,1))
        next_frames = np.transpose(self.frames[self.next_states[batchidx]], axes=(0,2,3,1))

        return {
            "reward": self.rewards[batchidx],
            "prev_state": prev_frames,
            "next_state": next_frames,
            "actions": self.actions[batchidx],
            "done_mask": self.is_done[batchidx]
        }
    
    def sample_mini_batch_wo_replacement(self, n):
        if(len(self.random_idx) == 0):
            self.random_idx = np.arange(self.counter)
            np.random.shuffle(self.random_idx)
        
        indices = self.random_idx[self.batch_num*n : min(self.counter - self.counter%n, (self.batch_num+1)*n)]
        if(len(indices) < n):
            indices = self.random_idx[0:64]
        
        prev_frames = np.transpose(self.frames[self.prev_states[indices]], axes=(0,2,3,1))
        next_frames = np.transpose(self.frames[self.next_states[indices]], axes=(0,2,3,1))
        
        if((self.batch_num+1)*n > self.counter):
            self.batch_num = 0
            np.random.shuffle(self.random_idx)
        else:
            self.batch_num += 1
        
        return {
            "reward": self.rewards[indices],
            "prev_state": prev_frames,
            "next_state": next_frames,
            "actions": self.actions[indices],
            "done_mask": self.is_done[indices]
        }
        

    def current_state(self):
        # assert not self.expecting_new_episode, "start new episode first"'
        assert self.frame_window is not None, "do something first"
        return self.frames[self.frame_window]

    def init_caches(self):
        self.rewards = np.zeros(self.capacity, dtype="float32")
        self.prev_states = -np.ones((self.capacity, self.num_frame_stack),
            dtype="int32")
        self.next_states = -np.ones((self.capacity, self.num_frame_stack),
            dtype="int32")
        self.is_done = -np.ones(self.capacity, "int32")
        self.actions = -np.ones(self.capacity, dtype="int32")

        # lazy to think how big is the smallest possible number. At least this is big enough
        self.max_frame_cache = self.capacity + 2 * self.num_frame_stack + 1
        self.frames = -np.ones((self.max_frame_cache,) + self.pic_size, dtype="float32")

class sqilExperienceHistory(ExperienceHistory):
    def __init__(self,
            num_frame_stack=4,
            capacity=int(1e5),
            pic_size=(96, 96),
            batch_ratio = 0.5,
            reward_func = None
    ):
        assert batch_ratio <= 1 and batch_ratio >= 0, "batch ratio between expert/learner should be between 0 and 1"
        self.batch_ratio = batch_ratio
        self.expertObsIndex = None
        self.reward_func = reward_func

        super().__init__(
            num_frame_stack=num_frame_stack,
            capacity=capacity,
            pic_size=pic_size)

    def loadExpertHistory(self, file_name):
        #TODO
        #remember index of the last expert frame
        #make reward equal to 1 for all obs

        startEpisode = True
        with open(file_name, 'rb') as f:
            demos = pickle.load(f)
        
        for demo in demos:
            for obs, act, done, rew in demo:
                if startEpisode:
                    frame_idx = self.counter % self.max_frame_cache
                    self.frame_window = np.repeat(frame_idx, self.num_frame_stack)
                    self.frames[frame_idx] = obs
                    self.expecting_new_episode = False
                    startEpisode = False
                else:
                    rew = 10 #SQIL expert rewards
                    super().add_experience(obs, act, done, rew)
                if done:
                    startEpisode = True
        self.expertObsIndex = self.counter % self.max_frame_cache
        print("Expert Data loaded with ", self.counter, " frames.", self.expertObsIndex)

    def check_learner_buffer_size(self, n):
        if(self.counter - self.expertObsIndex > n*10):
            return True
        else:
            return False

    def sample_mini_batch(self, n):
        #TODO
        #sample half from 0 to expert index and other half from expertindex to capacity
        
        expert_batch = int(n*self.batch_ratio)
        learner_batch = n - expert_batch


        count = min(self.capacity, self.counter)
        batchidx_expert = np.random.random_integers(0,self.expertObsIndex-1, expert_batch)
        batchidx_learner = np.random.random_integers(self.expertObsIndex, count-1, learner_batch)
        batchidx = np.append(batchidx_expert, batchidx_learner)

        prev_frames = self.frames[self.prev_states[batchidx]]
        next_frames = self.frames[self.next_states[batchidx]]

        return {
            "reward": self.rewards[batchidx],
            "prev_state": prev_frames,
            "next_state": next_frames,
            "actions": self.actions[batchidx],
            "done_mask": self.is_done[batchidx]
        }



    def add_experience(self, frame, action, done, reward):
        #TODO
        #just add to end of the buffer then, maybe we dont need an overload. 
        #make reward equal to 0 for all obs

        assert self.frame_window is not None, "start episode first"
        self.counter += 1
        
        if(self.counter == self.max_frame_cache): #not a good solution
            self.counter = self.expertObsIndex

        frame_idx = self.counter % self.max_frame_cache
        
        exp_idx = (self.counter - 1) % self.capacity

        self.prev_states[exp_idx] = self.frame_window
        self.frame_window = np.append(self.frame_window[1:], frame_idx)
        self.next_states[exp_idx] = self.frame_window
        self.actions[exp_idx] = action
        self.is_done[exp_idx] = done
        self.frames[frame_idx] = frame
        if(self.reward_func != None):
            self.rewards[exp_idx] = self.reward_func(self.counter)
        else:
            self.rewards[exp_idx] = 0 #SQIL
        if done:
            self.expecting_new_episode = True

if __name__ == "__main__":

    test_exp = sqilExperienceHistory(
                    num_frame_stack=3,
                    capacity=int(1e5),
                    pic_size=(96,96),
                    batch_ratio=0.5
                )
    expert_data_path = "expert_data/demos.pkl"
    test_exp.loadExpertHistory(expert_data_path)
    test_sample = test_exp.sample_mini_batch(64)
    #print(test_sample)