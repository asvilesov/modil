from mc_estimation import *
from model_dynamics import *
from data.experience_history import ExperienceHistory

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

data_buff = ExperienceHistory(num_frame_stack=4, capacity=int(2e5), pic_size=(96,96))
expert_data_path = "../data/expert/softDQNdemos.pkl"
data_buff.loadExpertHistory(expert_data_path)

with tf.device('/device:GPU:1'):
        '''Markov Chain'''
        mc_config = {  
                "loss": tf.keras.losses.MSE, 
                "optimizer": tf.keras.optimizers.Adam(learning_rate=1e-5),
                "epochs": 30,
                "batch_size": 64
                }
        mc_model = MarkovChainCNNModel(input_dim = (96,96, 4))
        trainMarkovChainCNN(model=mc_model, experience_history=data_buff, config=mc_config)

        print("done markov")

        '''Transition Probs'''
        trans_config = {  
                "loss": tf.keras.losses.MSE, 
                "optimizer": tf.keras.optimizers.Adam(learning_rate=1e-5),
                "epochs": 45,
                "batch_size": 64
                }
        trans = TransitionModel(input_dim=(96,96,4), action_dim=12)
        trainTransitions(model=trans, experience_history=data_buff, config=trans_config)

        print("done transitions")

        '''policy and train'''
        policy = CnnPolicy(action_dim=12)
        mc_est_config = {  
                "loss": tf.keras.losses.KLD, 
                "optimizer": tf.keras.optimizers.Adam(learning_rate=1e-5),
                "validation_split": 0.10,
                "max_obs": int(2e5),
                "epochs": 20,
                "batch_size": 64,
                "state_size": (96,96,4), 
                "action_size": 12
                }
        mc_est = imageMcEstimator(config=mc_est_config, mc_model=mc_model, dynamics_model=trans, policy=policy, dataset=data_buff)
        mc_est.train_mc()

        print("done policy")

policy.save("./models/cr_policy_next3")
# mc_model.save("./models/cr_mc")
# trans.save("./models/cr_trans")