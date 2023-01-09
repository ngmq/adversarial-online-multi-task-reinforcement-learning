import argparse
import time
import numpy as np
import pandas as pd
import math
from rlberry.envs import FiniteMDP, GridWorld
from rlberry.agents import AgentWithSimplePolicy
import constants

#import logging

import gym.spaces as spaces

#logger = logging.getLogger(__name__)


class RandomAgent(AgentWithSimplePolicy):
    name = "RandomAgent"

    def __init__(self, env, H1, H0 = 0, **kwargs):
        AgentWithSimplePolicy.__init__(self, env, **kwargs)
        self.H0 = H0
        self.H1 = H1
        self.H = H0 + H1

    def fit(self, budget=100, **kwargs):
        observation = self.env.reset()
        for ep in range(budget):
            action = self.policy(observation)
            observation, reward, done, _ = self.env.step(action)
            
    def policy(self, observation):
        return self.env.action_space.sample()  # choose an action at random
        
    def run_episode(self):
        state = self.env.reset() # reset environment
        phase1_reward = 0
        phase2_reward = 0
        # for h in range(self.H):
            # action = self.policy(state)
            # next_state, reward, done, _ = self.env.step(action)
            # state = next_state
            
            # if h <= self.H0:
                # phase1_reward += reward
            # else:
                # phase2_reward += reward    
        for h in range(self.H1):
            action = self.policy(state)
            next_state, reward, done, _ = self.env.step(action)
            state = next_state
            
            phase2_reward += reward                  
          
        return phase2_reward
        # return phase1_reward, phase2_reward
        
    def run_episode_with_initial_state(self, initial_state):
        state = initial_state
        self.env.setState(state)
        
        ep_reward = 0
        for h in range(self.H1):
            action = self.policy(state)
            next_state, reward, done, _ = self.env.step(action)
            state = next_state
            
            ep_reward += reward 

        return ep_reward
        
if __name__ == '__main__':    
    # arr = np.load("Results/RandomAgentALL.npy")
    # print(type(arr))
    # print(arr.shape)
    # print(arr)
    
    parser = argparse.ArgumentParser(description='Run ID')
    parser.add_argument("runid")
    args = parser.parse_args()
    print(args.runid)
    
    env_list, Lambda, GammaAll, GammaCover = create_environments3x3()
    M = len(env_list)
    print("M = {}".format(M))
    
    K = constants.K
    assert len(constants.seq) == K
    H0 = constants.H0
    H1 = constants.H1
    H = constants.H0 + constants.H1
    
    nseeds = 5
    seeds = range(0, nseeds)        
    
    start = time.time()
    run_data = np.zeros((K, 2))
    c_reward = 0
    c_reward_phase2 = 0
    
    # for k in range(K):
    for m in range(M):
        # m = constants.seq[k]
        # logger.info("k = {}, m = {}".format(k, m))
        randomagent = RandomAgent(env = env_list[m], H0 = H0, H1 = H1)    
        c_pertask_reward = 0
        c_pertask_reward_phase2 = 0        
        ep_reward1, ep_reward2 = randomagent.run_episode()     
        c_pertask_reward += ep_reward1 + ep_reward2
        c_pertask_reward_phase2 += ep_reward2
        
        c_pertask_reward /= nseeds
        c_pertask_reward_phase2 /= nseeds
        
        c_reward += c_pertask_reward
        c_reward_phase2 += c_pertask_reward_phase2
        #run_data[k, 0] = c_reward / (k+1)
        #run_data[k, 1] = c_reward_phase2 / (k+1)
        
       
        # np.save("Results/4x4/RandomAgent/Run{}/RandomAgentEp{}.npy".format(args.runid, k), run_data[k])
        # logger.info("k = {}, c_pertask_reward = {}, c_pertask_reward_phase2".format(k, c_pertask_reward, c_pertask_reward_phase2))
        # print("k = {}, m = {}, avg c_reward = {}, avg c_reward_phase2 = {}".format(k, m, c_reward / (k+1), c_reward_phase2 / (k+1)))        
        print("m = {}, ep_reward2 = {}".format(m, ep_reward2))
       
        
    # save all
    # np.save("Results/4x4/RandomAgent/Run{}/RandomAgentALL.npy".format(args.runid), run_data)
    end = time.time()
    print("Time = {}".format(end - start) + " seconds")
    