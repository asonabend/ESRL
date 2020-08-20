from utils import *

import argparse
import os
import sys






if __name__ == "__main__":

	# Load parameters
	parser = argparse.ArgumentParser()
	parser.add_argument("--env", default="Riverswim")     # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)             # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--episodes", default=300, type=int)  # Number of episodes in the training dataset 
	parser.add_argument("--risk_aversion", default=0.05, type=float)# Risk aversion parameter for ESRL
	parser.add_argument("--epsilon", default=0.1, type=float)  # Epsilon (noise) for the epsilon-greedy generation of training data
	parser.add_argument("--MDP_samples_train",default=250, type=int) # Number of MDP samples to use for training
	parser.add_argument("--MDP_samples_eval",default=500, type=int) # Number of MDP samples to use for OPPE
	args = parser.parse_args()
	

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if not os.path.exists("./models"):
		os.makedirs("./models")

	if not os.path.exists("./buffers"):
		os.makedirs("./buffers")

    ###
	epsilon,episodes,alpha = args.epsilon,args.episodes,args.risk_aversion
	env_name,train_K_no,eval_K_no,seed = args.env,args.MDP_samples_train, args.MDP_samples_eval,args.seed
    

	if env_name == 'Riverswim': # instantiate riverswim environment
		ep_length = 20
		env = riverSwim(episode_length=ep_length)
	else:
		ep_length = 1
		env = gym.make(env_name)    
	# State and action spaces            
	S_space = [i for i in range(env.observation_space.n)]
	A_space = [i for i in range(env.action_space.n)]

    ###
	print("---------------------------------------")	
	print(f"\nGenerating dataset, Env: {env_name}, Episodes: {episodes}, Epsilon: {epsilon},  Seed: {seed}\n")
	generate_dataset(epsilon,episodes,env_name,seed)
	data_dict = pickle.load( open("./buffers/obs_data"+env_name+"_seed_"+str(seed)+"_eps"+str(epsilon)+"_T_"+str(episodes)+".p", "rb" ) )
	episodes_repo_obs,H_tk_obs,mu_opt = data_dict['episodes_repo'],data_dict['H_tk'],data_dict['mu_st']
	pi_tsa,pi_st,visited_states = compute_pi_tsa(episodes_repo=episodes_repo_obs,S_space=S_space,A_space=A_space,tau=ep_length)
	print("---------------------------------------")       
	print(f"\nSetting: Training ESRL, Env: {env_name}, MDP samples: {train_K_no}, Risk aversion: {alpha},  Seed: {seed}\n")
	MV_Smu, _,_,_ = ESRL(H_tk_obs,alpha,ep_length,train_K_no,pi_st,pi_tsa,visited_states,S_space,A_space)
	print("---------------------------------------")       
	print(f"\nTesting on environment for 10000 episodes, Seed: {seed}\n")
	# test on envirnoment:
	true_rs = np.mean(test_policy(mu_st=MV_Smu,tau=ep_length,seed=seed))
	print("---------------------------------------")       
	print(f"\nOffline Policy Estimation (ESRL, IS & WIS), MDP samples: {eval_K_no}, Seed: {seed}\n")
	# value estimators:
	step_IS = step_IS_eval_mu(H_tk_obs,MV_Smu,pi_tsa,ep_length)[0]
	step_WIS = step_WIS_eval_mu(H_tk_obs,MV_Smu,pi_tsa,ep_length)[0]
	ESRL_V = np.mean(sample_Vs(pi_st,H_tk_obs,MV_Smu,eval_K_no,ep_length,A_space,S_space,seed))
	print("---------------------------------------")
	print(f"\nMean Reward:, Online test: {np.round(true_rs,3)}, ESRL_V: {np.round(ESRL_V,3)}, IS: {np.round(step_IS,3)}, WIS: {np.round(step_WIS,3)}, Seed: {seed}\n")
    #Store results in dictionary    
	results_seed_epi_eps_alph = {}
	results_seed_epi_eps_alph[(seed,episodes,epsilon,alpha)] = {'true_rs':true_rs,'step_IS':step_IS,'step_WIS':step_WIS,'ESRL_V':ESRL_V}
	
    # If no dictionary is stored store current dict otherwise append to existing dict

	if not os.path.exists("./results/results_seed_epi_eps_alpha"+env_name+".p"):
		pickle.dump(results_seed_epi_eps_alph, 
            open( "./results/results_seed_epi_eps_alpha"+env_name+".p", "wb" ) )
	else:
		results_dict = pickle.load( open("./results/results_seed_epi_eps_alpha"+env_name+".p", "rb" ) )
		results_dict[(seed,episodes,epsilon,alpha)] = results_seed_epi_eps_alph[(seed,episodes,epsilon,alpha)]
		pickle.dump(results_dict, 
            open( "./results/results_seed_epi_eps_alpha"+env_name+".p", "wb" ) )
