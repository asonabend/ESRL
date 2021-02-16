import gym
from gym import spaces
import random
import collections
import numpy as np
import pickle
from collections import namedtuple
import pandas as pd
import os
from tqdm import tqdm

import sys

#import matplotlib.pyplot as plt
###########################################################################
################  openAI wrapper for Riverswim environment ################
###########################################################################

class riverSwim(gym.Env):
    """Riverswim Environment that follows gym interface"""
    #metadata = {'render.modes': ['human']}
    def __init__(self,episode_length):
        super(riverSwim, self).__init__()
        self.time = 0
        self.state = 0
        self.episode_length = episode_length
        self.done = False
        self._max_episode_steps = episode_length
        # Define action and observation space
        N_DISCRETE_ACTIONS = 2
        N_DISCRETE_STATES = 6 
        #N_TIME_STEPS = episode_length
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Discrete((N_DISCRETE_STATES))
        self.observation_dim = 2
    # Execute one time step within the environment
    def step(self,action):
        if action not in [0,1]:
            return 'error'        
        if self.time == self.episode_length:
            self.done = True
        self.time += 1
        self.reward = 0 # no reward
        if self.state == 0:
            if action == 1: # swim to the right
                if np.random.binomial(1,.6)==1: # w.p. = 0.6 get to the right otherwise stay in state = 0
                    self.state = 1                
            else: #action == 0, stays in state = 0 and has the reward 5/1000
                self.reward = 5/1000
        elif self.state == 5:
            if action == 1: # swim to the right
                if np.random.binomial(1,.6)==1: # w.p. = 0.6 swim succesfully to the right 
                    self.reward = 1
                else: # w.p. 0.4 current takes it to the left
                    self.state = 4
            else: # action == 0
                self.state = 4
        else: #states 1,2,3,4
            if action == 1: # swim to the right
                dice = np.random.choice(3, 1, p=[.05,0.6,0.35])
                if dice==0: # w.p. = 0.05 current takes it to the left
                    self.state -= 1
                elif dice==2: # w.p. = 0.6 current it stays in the same state, w.p. 0.35 gets to the right
                    self.state += 1
            else: # action == 0
                self.state -= 1
        info = 'na'
        return self.state,self.reward,self.done,info
    def reset(self):
    # Reset the state of the environment to an initial state
        self.time = 0
        self.state = 0
        self.done = False
        return self.state

################################################
################# Test Policy Online ###########
################################################


## test a policy on Riverswim
def test_policy(mu_st,tau,seed,epsilon=0):
    np.random.seed(seed)
    test_episodes = 10000
    test_episodes_repo = experienceRepository(test_episodes)
    # instantiate riverswim environment:
    env = riverSwim(episode_length=tau)
    #### OPPSRL Algorithm 
    rewrds_per_ep = []
    for k in tqdm(range(int(test_episodes))):
        state = env.reset()
        state_seq = state
        reward_seq,action_seq = -100,-100 
        for t in range(tau):
            if np.random.binomial(1,epsilon)==1: # w.p. epsilon choose random action
                action = int(np.random.choice(2, 1, p=[0.5,0.5]))
            else:
                action = mu_st[(state,t)] # w.p. 1-epsilon choose mu_k(s)
            next_state, reward, done, _ = env.step(action)
            # store episode
            reward_seq = np.vstack((reward_seq,reward))
            action_seq = np.vstack((action_seq,action))  
            state = next_state
            state_seq = np.vstack((state_seq,state))
        # store episode in repo
        rewrds_per_ep.append(sum(reward_seq[1:]))
        test_episodes_repo.store(state_seq[:-1], action_seq[1:],[],reward_seq[1:])
    # Compute H_tk:
    test_episodes = test_episodes_repo.memory
    test_samp = Transition(*zip(*test_episodes))
    test_H_tk = build_H_tk(test_samp,k)
    all_rewards = test_H_tk[:,2]
    return rewrds_per_ep

################################################
################## OPPE functions ##############
################################################

# Step importance sampling evaluation

def step_IS_eval_mu(H_T,mu_st,pi_tsa,tau):
    # H_t's columns are [state,action,reward,next state]
    V_mu,curr_V = [],0
    for t in range(H_T.shape[0]):
        if t % tau == 0:
            V_mu.append(curr_V)
            curr_V,rho = 0,1
        s,a,r = H_T[t,0],H_T[t,1],H_T[t,2]
        if (t % tau,s,a) in pi_tsa.keys():
            rho *= (a==mu_st[(s,t % tau)])/pi_tsa[(t % tau,s,a)]
        else: 
            rho = 1
        curr_V += rho*r
    V_mu.append(curr_V)
    return np.mean(V_mu), np.std(V_mu)

# Weighted step importance sampling evaluation

def step_WIS_eval_mu(H_T,mu_st,pi_tsa,tau):
    # H_t's columns are [state,action,reward,next state]
    # Compute denominator for weighted importance sampling:
    T = H_T.shape[0]//tau
    rho_s,i = {i:[1] for i in range(T)},-1
    for t in range(H_T.shape[0]):
        if t % tau == 0:
            i += 1
        s,a,r = H_T[t,:3]
        if (t % tau,s,a) in pi_tsa.keys():
            rho_s[i].append(rho_s[i][-1]*(a==mu_st[(s,t % tau)])/pi_tsa[(t % tau,s,a)])
        else: 
            rho = 1
    # H_t's columns are [state,action,reward,next state]
    V_mu,curr_V = [],0
    for t in range(H_T.shape[0]):
        if t % tau == 0:
            V_mu.append(curr_V)
            curr_V,rho = 0,1
        s,a,r = H_T[t,0],H_T[t,1],H_T[t,2]
        if (t % tau,s,a) in pi_tsa.keys():
            w_t = np.mean([rho_s[i][t % tau] for i in range(T)])
            rho *= (a==mu_st[(s,t % tau)])/pi_tsa[(t % tau,s,a)]
        else: 
            rho = 1
        if w_t == 0:
            curr_V += rho*r
        else:
            curr_V += rho*r/w_t
    V_mu.append(curr_V)
    return np.mean(V_mu), np.std(V_mu)

# ESRL OPPE APPROACH

# Computes value function samples at state s by sampling MDP's and running policy on them
def sample_Vs(pi_sa,H_T,mu_st,samps_No,tau,A_space,S_space,seed,sepsis=False,epsilon=0):    
    np.random.seed(seed)
    V_samps = np.zeros((samps_No))
    # Sample MDP from posterior based on IPW repo:
    R_sa_dict, P_sas_dict = sampleK_MDPs(S_card=len(S_space),A_space=A_space,H_tk=H_T,K=samps_No,sepsis=sepsis)#,pi_sa=pi_sa)
    for i in range(samps_No):
        # Sample MDP from posterior based on IPW repo:
        R_sa, P_sas = R_sa_dict[i], P_sas_dict[i]
        # initialize value V and policy function dictionaries:
        if sepsis:
            start_s = np.where(H_T[:,3]==-1)[0]+1 # index of initial states rows
            s = int(np.random.choice(H_T[start_s[:-1],0]))
        else: 
            s = 0 #all episodes start at state = 0 in riverswim
        V_s0 = 0
        weights = [1]
        for t in range(tau):
            # Choose action a according to (random) policy    
            if np.random.binomial(1,epsilon)==1: # w.p. epsilon choose random action
                a = np.random.choice(A_space)
            else:
                a = mu_st[(s,t)] # w.p. 1-epsilon choose mu_k(s)
            sum_pi = 1#sum([np.product(weights)/pi_sa[(s,a)] for a in A_space])
            #weights.append(1/pi_sa[(s,a)])            
            V_s0 += R_sa[(s,a)]*np.product(weights)/sum_pi
            # Sample an action according to the sampled MDP and action taken
            s = int(np.random.choice(len(S_space),1, p=P_sas[(s,a)]))
        V_samps[i] = V_s0
    return V_samps

################################################
################# Generate dataset #############
################################################
#In the next cell we generate a kind of experience replay repository with different attributes:

#`store`: stores the observation tuple $(s_i,a_i,s_i',r_i)$

#`sample`: generates a sample of size `batch_size`

#`len`: function that returns the amount of samples in the repository

#### experience replay repo ####
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class experienceRepository(object):

    def __init__(self, limit):
        self.limit = limit
        self.memory = []
        self.position = 0

    def store(self, *args):
        """Stores a transition."""
        if len(self.memory) < self.limit:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.limit

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

################################################
################# Sampling MDPs ################
################################################

# Sample K MDPs simultaneously 
def sampleK_MDPs(S_card,A_space,H_tk,K,pi_sa=None,sepsis=False):
    if sepsis:
        prior_par = {'m0':0,'lamb0':1e+3,'alpha0':5.01,'gamma0':1}
    else:
        prior_par = {'m0':0,'lamb0':1,'alpha0':1.01,'gamma0':1}    # Riverswim
        #prior_par = {'m0':1,'lamb0':.01,'alpha0':1.1,'gamma0':1}    # Frozen lake
    #prior_par = {'m0':1,'lamb0':1,'alpha0':1.01,'gamma0':1}
    #prior_par = {'m0':0,'lamb0':1e4,'alpha0':5.01,'gamma0':1}
    #{'m0':0,'lamb0':1e+4,'alpha0':5.01,'gamma0':1}
    all_states = [float(i) for i in range(S_card)]
    P_sas_dict,R_sa_dict = {k:{} for k in range(K)},{k:{} for k in range(K)}
    for a in A_space:
        for s in all_states:            
            H_sa=None
            dirich_alpha = [1/S_card]*S_card # reset state-action probabilities to 1/alpha
            if H_tk is not None:                
                indx = np.where(np.product(np.array([s,a])==H_tk[:,:2],axis=1)==1)[0]
                if indx.shape[0] > 0:           
                    ## Check if we should use informative prior:
                    #if pi_sa is not None:
                        #prior_par['m0'],prior_par['lamb0'] = pi_sa[(s,a)],len(indx)
                    H_sa = H_tk[indx,:]
                    # update dirichlet's alphas with current counts
                    for nxt_s in np.unique(H_sa[:,3])[1:]:# all next states (remove -inf)
                        dirich_alpha[int(nxt_s)] += sum(nxt_s == H_sa[:,3]) # add counts for that (s,a) pair to the Dirich. parameter
            # sample means for the reward distributions        
            K_Rs = normal_gamma_sample(m0=prior_par['m0'],lamb0=prior_par['lamb0'],alpha0=prior_par['alpha0'],gamma0=prior_par['gamma0'],H_sa=H_sa,K=K)
            #prior_par = {'m0':1,'lamb0':1,'alpha0':1.0001,'gamma0':1}
            # Draw random vector of probabilities from Dir. posterior for the transition distribution:
            K_P_sas = np.random.dirichlet(dirich_alpha,K)
            for k in range(K):
                R_sa_dict[k][(s,a)] = K_Rs[k]
                P_sas_dict[k][(s,a)] = K_P_sas[k]
    return R_sa_dict, P_sas_dict

# sample a posterior distributon for the parameters on the state action pair (s,a)
def normal_gamma_sample(m0,lamb0,alpha0,gamma0,H_sa=None,K=1):
    m,lamb,alpha,gamma=m0,lamb0,alpha0,gamma0
    if H_sa is not None: 
        n_sa = H_sa.shape[0]
        r_bar = np.mean(H_sa[:,2])
        r_sq_bar = np.mean(H_sa[:,2]**2)
        m = (lamb0*m0+n_sa*r_bar)/(lamb0+n_sa)
        lamb = lamb0 + n_sa
        alpha = alpha0 + n_sa/2
        gamma = gamma0 + 0.5*n_sa*(r_sq_bar-r_bar**2)+(n_sa*lamb0*(r_bar-m0)**2)/(2*(lamb0+n_sa))
    #else:
        #m,lamb,alpha,gamma=m0,lamb0,alpha0,gamma0
    tautau = np.random.gamma(alpha, gamma, K)
    sigma = 1/(lamb*tautau)      
    mu_sa = np.random.normal(m, sigma, K)
    return mu_sa

##################################################
################# Generating Buffers ################
#####################################################


### Train an expert policy using PSRL Algorithm (Osband et. al. 2013):
def train_pi(env_name):
    if env_name == 'Riverswim': # instantiate riverswim environment
        ep_length = 20
        env = riverSwim(episode_length=ep_length)
    else:
        ep_length = 1
        env = gym.make(env_name)    
    # State and action spaces            
    S_space = [i for i in range(env.observation_space.n)]
    A_space = [i for i in range(env.action_space.n)]

    # instantiate memory:
    episodes_repo = experienceRepository(10000)
    episode_rewards = []
    H_tk=None
    for k in tqdm(range(int(1e3))):
        # Sample MDP:
        R_sa, P_sas = sampleK_MDPs(S_card=len(S_space),A_space=A_space,H_tk=H_tk,K=1)
        R_sa, P_sas = R_sa[0], P_sas[0]        
        # Compute mu_k
        muK_st = compute_policy(R_sa,P_sas,A_space,S_space,ep_length)
        state = env.reset()
        state_seq = state
        reward_seq,action_seq = -100,-100 
        loop_len = ep_length if env_name == 'Riverswim' else 1000
        for t in range(loop_len):
            s_ext = t if env_name == 'Riverswim' else 0
            action = muK_st[(state,s_ext)]
            next_state, reward, done, _ = env.step(action)
            # store episode
            reward_seq = np.vstack((reward_seq,reward))
            action_seq = np.vstack((action_seq,action))  
            state = next_state
            state_seq = np.vstack((state_seq,state))
            # Check if done (Riverswim is done at t=ep_length)
            if env_name != 'Riverswim' and done:
                episode_rewards.append(reward)
                env.reset()
                break
            # store episode in repo
        episodes_repo.store(state_seq[:-1], action_seq[1:],[],reward_seq[1:])
        # Compute H_tk for posterior:

        if k == 0:
            H_tk = np.hstack((state_seq[:-1],action_seq[1:],reward_seq[1:],np.vstack((state_seq[2:],float("-inf")))))
        else:
            sars_pairs = np.hstack((state_seq[:-1],action_seq[1:],reward_seq[1:],np.vstack((state_seq[2:],float("-inf")))))
            H_tk = np.vstack((H_tk,sars_pairs))

        ##
        
            
        #if k % 500==0:
            #print(k,np.mean(episode_rewards[-500:]))
            
            #plt.plot([np.mean(episode_rewards[:i])  for i in range (len(episode_rewards))])
            #plt.plot([np.mean(H_tk[:i,2])  for i in range (len(H_tk))])
            #plt.show()
            #print(np.mean(episode_rewards[-500:]))
            #R_sa, P_sas = sampleK_MDPs(S_card=len(S_space),A_space=A_space,H_tk=H_tk,K=500)
            #plt.hist([R_sa[k][(0,0)] for k in range(500)], color = 'blue', edgecolor = 'black',bins =40)
            #plt.show()

    pickle.dump({'R_sa':R_sa,'P_sas':P_sas,'muK_st':muK_st}, 
            open( './models/'+env_name+'_PSRL_optPolicy.p', 'wb' ) )

## Build repository
def build_H_tk(samp,epi_no):   
    # build matrix of observations with concatenated episods, whith rows of the form: [state, action, reward]
    tt = 0
    H_tk = np.hstack((samp.state[tt],samp.action[tt],samp.reward[tt],np.vstack((samp.state[tt][1:],float("-inf")))))
    for tt in range(1,epi_no):
        next_state = np.vstack((samp.state[tt][1:],float("-inf")))
        sars_pairs = np.hstack((samp.state[tt],samp.action[tt],samp.reward[tt],next_state))
        H_tk = np.vstack((H_tk,sars_pairs))
    return H_tk


#### Generate a dataset by exploring environment with PSRL trained policy

def generate_dataset(epsilon,episodes,env_name,seed):
    # function to generate a dataset according to a mixture of a trained policy mu_k from PSRL and a random policy and
    # at any state s the decision is made according to: mu_k(s) w.p. 1-epsilon, rndm{0,1} w.p. epsilon
    # load (and generate and load) PSRL trained policy
    if not os.path.exists('./models/'+env_name+'_PSRL_optPolicy.p'):
        print('Training PSRL behavior policy...')
        sys.stderr.write('Training expert policy with PSRL')
        np.random.seed(116687)
        train_pi(env_name)
    # Load behavior policy
    PSRL_dict = pickle.load(open('./models/'+env_name+'_PSRL_optPolicy.p', "rb" ) ) 
    _,_,mu_st = PSRL_dict['R_sa'],PSRL_dict['P_sas'],PSRL_dict['muK_st']

    # instantiate memory:
    obs_episodes_repo = experienceRepository(episodes)
    # instantiate riverswim environment:
    if env_name == 'Riverswim': # instantiate riverswim environment
        ep_length = 20
        env = riverSwim(episode_length=ep_length)
    A_No = env.action_space.n
    np.random.seed(seed) 
    #### exploration
    for k in range(int(episodes)):
        # Compute mu_k
        state = env.reset()
        state_seq = state
        reward_seq,action_seq = -100,-100 
        for t in range(ep_length):
            if np.random.binomial(1,epsilon)==1: # w.p. epsilon choose random action
                action = int(np.random.choice(A_No, 1))
            else:
                action = mu_st[(state,t)] # w.p. 1-epsilon choose mu_k(s)
            next_state, reward, done, _ = env.step(action)
            # store episode
            reward_seq = np.vstack((reward_seq,reward))
            action_seq = np.vstack((action_seq,action))  
            state = next_state
            state_seq = np.vstack((state_seq,state))
            # store episode in repo
        obs_episodes_repo.store(state_seq[:-1], action_seq[1:],state_seq[1:],reward_seq[1:])
    #if k % 1000==0:    
    # Compute H_tk:
    obs_episodes = obs_episodes_repo.memory
    obs_samp = Transition(*zip(*obs_episodes))
    obs_H_tk = build_H_tk(obs_samp,k)
    #print(k,np.mean(obs_H_tk[:s,2]))s
    
    pickle.dump({'episodes_repo':obs_episodes_repo,'H_tk':obs_H_tk,'mu_st':mu_st}, 
            open( "./buffers/obs_data"+env_name+"_seed_"+str(seed)+"_eps"+str(epsilon)+"_T_"+str(episodes)+".p", "wb" ) )

################################################
################# ESRL ################
################################################

# Compute the mode
def mode(ls):
    # calculate the frequency of each item
    data = collections.Counter(ls)
    data_list = dict(data)
    max_cnt = max(list(data_list.values()))
    mode_val = [num for num, freq in data_list.items() if freq == max_cnt]
    return mode_val

# compute policy for the ESRL algorithm
def compute_policy(R_sa,P_sas,A_space,S_space,ep_length,rtrn_v=False):
    # initializa value Vt(S) and policy function dictionaries:
    mu_st = {}
    V_st = {(s,ep_length):0 for s in S_space}
    for t in range(ep_length-1,-1,-1):
        for s in S_space:
            q_vals = [R_sa[(s,a)] + sum([P_sas[(s,a)][int(nxt_s)]*V_st[(nxt_s,t+1)] for nxt_s in S_space]) for a in A_space]
            V_st[(s,t)],mu_st[(s,t)] = np.max(q_vals),np.argmax(q_vals)
    if rtrn_v:
        return mu_st,V_st
    else:
        return mu_st
#### Computes the null probability
def P_H0_MV(s,t,a_behavior,a_mu,Mk_R_sa,Mk_P_sas,kset,V_st,visited_states,pi_tsa,S_space,A_space):
    Qs,i = np.zeros((len(kset),len(A_space))),0
    for k in kset:
        # compute Q values for current state of interest            
        R_sa,P_sas = Mk_R_sa[k],Mk_P_sas[k]
        Qs[i,:] = [(R_sa[(s,a)] + sum([P_sas[(s,a)][int(nxt_s)]*V_st[k][(nxt_s,t+1)] for nxt_s in S_space])) for a in A_space]
        i += 1
    return np.mean(Qs[:,a_mu]<Qs[:,a_behavior]),Qs

#### Estimates behavior policy and propensity scores for IS weights (assigns None to (s,a) pairs not in observed data)
def compute_pi_tsa(episodes_repo,S_space,A_space,tau,sepsis=False,H_T=None):
    if not sepsis:
        episodes = episodes_repo.memory
        samp = Transition(*zip(*episodes))
        H_T = build_H_tk(samp,len(episodes_repo))        
        # augment history with column of episode stage
        # [time stage,state,action,reward,next state]
        H_T = np.hstack((np.array([[i for i in range(tau)]*(H_T.shape[0]//tau)]).T,H_T))
    T = H_T.shape[0]
    all_states,all_actions = S_space,A_space
    pi_tsa = {}
    visited_states = {t:set() for t in range(tau+1)}
    for a in all_actions:
        for s in all_states:
            for t in range(tau):
                # times action a was selected in state s at stage t
                indx_tsa = np.where(np.product(np.array([t,s,a])==H_T[:,:3],axis=1)==1)[0]
                # times state s was observed at stage j
                indx_ts = np.where(np.product(np.array([t,s])==H_T[:,:2],axis=1)==1)[0]
                if len(indx_ts)>0:
                    visited_states[t].add(s)
                    if len(indx_tsa)>0:
                        pi_tsa[(t,s,a)] = (len(indx_tsa)/len(indx_ts))
                    else:
                        pi_tsa[(t,s,a)] = .001
                else:
                    pi_tsa[(t,s,a)] = None

    # Compute behavior policy base on most likely action
    pi_st = {}
    for t in range(tau-1,-1,-1):
        for s in S_space:
            if s in visited_states[t]:    
                pi_st[(s,t)] = np.argmax([pi_tsa[(t,s,a)] for a in A_space])
            else: 
                pi_st[(s,t)] = int(np.random.choice(len(A_space), 1))
    return pi_tsa,pi_st,visited_states

# ESRL Algorithm: 
def ESRL(H_T,alpha,tau,K_no,pi_st,pi_tsa,visited_states,S_space,A_space,sepsis=False):
    # Generate sets for estimating Q and testing H0
    K_ls = list(range(K_no))
    I_1,I_2 = K_ls[:len(K_ls)//2],K_ls[len(K_ls)//2:]
    # Samples K MDPs from posterior s
    Mk_R_sa, Mk_P_sas = sampleK_MDPs(S_card=len(S_space),A_space=A_space,H_tk=H_T,K=K_no,sepsis=sepsis)
    # initialize value Vtau(S) and policy function dictionaries:
    V_st = {k:{(s,tau):0 for s in S_space} for k in range(K_no)}
    mu_st_alpha,mu_st,maj_vote_mu,maj_vote_mu_alpha,maj_vote_set_alpha = {k:{} for k in range(K_no)},{k:{} for k in range(K_no)},{},{},{}
    Qs_st = {}
    for t in range(tau-1,-1,-1):
        for s in S_space:
            for k in range(K_no):
                ##
                # compute Q values for current state of interest            
                R_sa,P_sas = Mk_R_sa[k],Mk_P_sas[k]
                q_vals = [(R_sa[(s,a)] + sum([P_sas[(s,a)][int(nxt_s)]*V_st[k][(nxt_s,t+1)] for nxt_s in S_space])) for a in A_space]
                # Compute mu_k
                mu_st[k][(s,t)] = np.argmax(q_vals)
            # Compute policy based on majority vote:            
            maj_vote_mu[(s,t)] = int(mode([mu_st[k][(s,t)] for k in I_1])[0])
            # Compute P(H_0|s,d,H_T)
            P_0,Qs_st[(s,t)] = P_H0_MV(s,t,a_behavior=pi_st[(s,t)],a_mu=maj_vote_mu[(s,t)],Mk_R_sa=Mk_R_sa,Mk_P_sas=Mk_P_sas,kset=I_2,V_st=V_st,visited_states=visited_states,pi_tsa=pi_tsa,S_space=S_space,A_space=A_space)
            for k in range(K_no):
                # Compute policy based on P-value rule
                mu_st_alpha[k][(s,t)] = mu_st[k][(s,t)] if P_0<alpha else pi_st[(s,t)] #if P_0>= alpha   
                # Compute value function based on chosen policy
                V_st[k][(s,t)] = float(*[(R_sa[(s,a)] + sum([P_sas[(s,a)][int(nxt_s)]*V_st[k][(nxt_s,t+1)] for nxt_s in S_space])) for a in [mu_st_alpha[k][(s,t)]]])
            # Compute policy based on majority vote, and set of k's which chose the most common action:            
            maj_vote_mu_alpha[(s,t)] = int(mode([mu_st_alpha[k][(s,t)] for k in I_1])[0])        
            maj_vote_set_alpha[(s,t)] = [k for k in I_1 if maj_vote_mu_alpha[(s,t)] == mu_st_alpha[k][(s,t)]]
    # Define majority voting set and check if there are models in all:
    MV_set = set(k for k in range(K_no))
    for key in maj_vote_set_alpha.keys():
        MV_set = MV_set.intersection(maj_vote_set_alpha[key])
    if len(MV_set)>0:
        chosen_k = np.random.choice(list(MV_set))
    else:    
        chosen_k = int(mode([k for key in list(maj_vote_set_alpha.keys()) for k in maj_vote_set_alpha[key]])[0])
    return mu_st_alpha[chosen_k], Mk_R_sa[chosen_k],Mk_P_sas[chosen_k],Qs_st
