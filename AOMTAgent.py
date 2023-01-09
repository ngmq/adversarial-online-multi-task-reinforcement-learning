import time
import numpy as np
import pandas as pd
import math
from copy import deepcopy
from rlberry.envs import GridWorld
from rlberry.agents import AgentWithSimplePolicy
from UCBVICHAgent import UCBVICHAgent
from ExploreIDAgent import ExploreIDAgent


class AOMTAgent():
    name = "ExploreID"
    
    def __init__(self,
                 env,
                 H0,
                 H1,
                 Need,
                 Gamma,
                 nepisodes,
                 M,
                 failure_prob=0.3,                 
                 **kwargs):
        
        assert env is not None and isinstance(env, GridWorld)
        assert H0 is not None and H1 is not None and Need is not None and Gamma is not None and M is not None
        self.exploreAgent = ExploreIDAgent(env = env, horizon = H0, Need = Need,  Gamma = Gamma)   
        self.ucbviAgent = UCBVICHAgent(env = env, horizon = H1,  nepisodes = nepisodes, M = M, failure_prob = failure_prob / 3)
        
        assert id(env) != id(self.exploreAgent.env) and id(self.exploreAgent.env) != id(self.ucbviAgent.env)
        # self.ModelClusters = list()
        
    # def reset(self, **kwargs):        
        # shape_sa = (self.S, self.A)
        # shape_sas = (self.S, self.A, self.S)

        # # (s, a) visit counter
        # self.N_sa = np.zeros(shape_sa)
        # self.N_sas = np.zeros(shape_sas)
        
        # self.P_hat = np.zeros(shape_sas) # start from all zeros, not uniform distribution      
    def reseed(self, seed):
        self.exploreAgent.reseed(seed)
        self.ucbviAgent.reseed(seed)
        
    def policy(self, observation):
        pass
    
    def run_exploration(self):
        isSucc, explore_reward = self.exploreAgent.run_episode()
        
        return isSucc, explore_reward
    
    def run_optimization(self):
        current_state = self.exploreAgent.env.state
        phase2_reward = self.ucbviAgent.run_episode(initial_state = current_state)
        return phase2_reward
        
    def fit(self):    
        pass