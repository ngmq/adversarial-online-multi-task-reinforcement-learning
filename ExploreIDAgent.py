import time
import numpy as np
import pandas as pd
import math
from copy import deepcopy

from rlberry.envs import FiniteMDP, GridWorld
from rlberry.agents import AgentWithSimplePolicy
import utils
import constants
# import logging

import gym.spaces as spaces

# logger = logging.getLogger(__name__)

class ExploreIDAgent(AgentWithSimplePolicy):
    """
    Parameters
    ----------
    env : gym.Env
        Environment with discrete states and actions.
    horizon : int
        Horizon of the objective function.
    Gamma : dictionary of sets
        The distinguishing set. Gamma[s] = {a1, a2, ...} the set of a such that (s,a) is needed
    """
    name = "ExploreID"
    
    def __init__(self,
                 env,
                 horizon,
                 Need,
                 Gamma=None,
                 copy_env = True,
                 **kwargs):
        # init base class        
        AgentWithSimplePolicy.__init__(self, env, copy_env = copy_env, **kwargs)
        
        assert isinstance(env, GridWorld)
        
        if horizon is None:
            assert "No horizon is given."
        if Need is None:
            assert "No need is given."
        
        self.S = self.env.observation_space.n
        self.A = self.env.action_space.n
        self.H = horizon        
        self.N = Need      
        
        if Gamma is not None and len(Gamma) > 0:
            self.Gamma = deepcopy(Gamma)
        else:
            # visit every state-action pairs if Gamma is not specified
            self.Gamma = {}
            for s in range(self.S):
                self.Gamma[s] = set()
                self.Gamma[s].update(range(self.A))
        
        # check environment
        assert isinstance(self.env.observation_space, spaces.Discrete)
        assert isinstance(self.env.action_space, spaces.Discrete)
            
        # initialize
        self.reset()
    
    # def resetEnv(self, env, initial_state):
        # assert self.S == env.observation_space.n
        # assert self.A == env.action_space.n        
        # self.env = env
        # self.env.setState(initial_state)
        
    def reset(self, **kwargs):        
        shape_sa = (self.S, self.A)
        shape_sas = (self.S, self.A, self.S)

        # (s, a) visit counter
        self.N_sa = np.zeros(shape_sa)
        self.N_sas = np.zeros(shape_sas)
        
        self.P_hat = np.zeros(shape_sas) # start from all zeros, not uniform distribution
        
    def _update(self, state, action, next_state):
        self.N_sa[state, action] += 1
        self.N_sas[state, action, next_state] += 1

        nn = self.N_sa[state, action]        
        prev_p = self.P_hat[state, action, :]
        
        self.P_hat[state, action, :] = (1.0 - 1.0 / nn) * prev_p
        self.P_hat[state, action, next_state] += 1.0 / nn  

        # update Gamma
        if nn >= self.N and (state in self.Gamma):
            self.Gamma[state].discard(action)            
    
    def policy(self, observation):
        pass
        
    def _get_action(self, state):
        """ Sampling policy. """        
        assert self.Gamma is not None        
        if (self.Gamma.get(state) is not None) and len(self.Gamma[state]) > 0:
            tmp = 0
            amax = 0
            for a in self.Gamma[state]:
                if self.N_sa[state, a] >= tmp:
                    amax = a
                    tmp =  self.N_sa[state, a]
            return amax
        else:
            tmp = 0.0
            amax = -1
            for a in range(self.A):
                Sum = 0.0
                for sprime in range(self.S): # replace by neighbors later
                #for sprime in self.env.valid_neighbors[state]:
                    if (self.Gamma.get(sprime) is not None) and (len(self.Gamma[sprime]) > 0):
                        Sum += self.P_hat[state, a, sprime]
                if (not math.isclose(Sum, 0.0)) and Sum > tmp:
                    amax = a
                    tmp = Sum
            if amax == -1:
                amax = self.env.action_space.sample()
            return amax
        
    def run_episode_with_initial_state(self, initial_state):       
        # interact for H steps        
        state = initial_state
        # self.env.setState(state)
        self.env.state = state
        
        # history = {}
        explore_reward = 0.0
        for h in range(self.H):
            action = self._get_action(state)
            # logger.info("action is {}".format(action))
            next_state, reward, done, _ = self.env.step(action)
            explore_reward += reward
            
            self._update(state, action, next_state)
            # if history.get((state, action, next_state)) is None:
                # history[(state, action, next_state)] = 0
            
            # history[(state, action, next_state)] += 1

            state = next_state           
        
        # return sum of rewards collected in the episode        
        isSucc = True
        for s in range(self.S):
            if self.Gamma.get(s) is not None:                
                if len(self.Gamma[s]) > 0:
                    got = [self.N_sa[s, a] for a in self.Gamma[s]]
                    print("Failed at s = {}, actions = {}, only got {}".format(s, self.Gamma[s], got))
                    isSucc = False
        return isSucc, explore_reward

    def run_episode(self):
        state = self.env.reset()
        return self.run_episode_with_initial_state(state)
        
    def fit(self):    
        pass


if __name__ == '__main__':
    np.random.seed(0)
    
    # S = 9
    # A = 4

    # R = np.random.uniform(0, 1, (S, A))
    # P = np.random.uniform(0, 1, (S, A, S))
    # initial_state_distr = 0  # np.ones(S)/S
    # for ss in range(S):
        # for aa in range(A):
            # P[ss, aa, :] /= P[ss, aa, :].sum()

    # env = FiniteMDP(R, P, initial_state_distribution=initial_state_distr)
    # # env.reseed(233) # this one has no effects. The environment is reseeded by AgentWithSimplePolicy.__init__(self, env, **kwargs)
    # Gamma = None
     
    # nepisodes = 2
    # count = 0
    # for k in range(nepisodes):
        # initial_state = env.reset() #stochastic setting
        # # logger.info(type(initial_state)) # <class 'int'>
        # exploreIdagent = ExploreIDAgent(env, horizon = 1000, Need = 41) # set Need = 42 got 9 successes out of 10 episodes
        # exploreIdagent.reseed(233)
        # succ = exploreIdagent.run_episode(initial_state)
        # if succ is True:
            # count += 1               
        
    # logger.info("Out of {} episodes, {} were successful".format(nepisodes, count))
    
    env_list, Lambda, GammaAll, GammaCover = utils.create_environments(4)
    M = len(env_list)
    A = env_list[0].A # 4 up down left right
    S = env_list[0].S    
    # assert(M == constants.M and A == constants.A and S == constants.S)
    K = constants.K
    p1 = constants.p1
    logTerm = utils.computeLogTerm(K, len(GammaCover), p1)
    N = utils.computeN(Lambda, logTerm)
    D = constants.D
    n = utils.computeNumberOfIntervals(N, logTerm)
    
    Gamma = dict()
    for s, a in GammaCover:
        if s not in Gamma:
            Gamma[s] = set()
        Gamma[s].add(a)
        
    if Gamma is None or len(Gamma) == 0:
        Gamma = dict()
        for s in range(S):
            Gamma[s] = set() 
            for a in range(A):
                Gamma[s].add(a)
                
    H0 = int(2 * D * len(GammaCover) * n)
     # H0 = int(D * S * A * N)
    # H0 = constants.H0
    
    print("N = {}, H0 = {}".format(N, H0))
    # print("Gamma = {}".format(Gamma))
    
    countSucc = 0
    for _ in range(5):
        for m in range(M):
        #for _ in range(3):
            start = time.time()
            env = env_list[m]
        
            initial_state = env.reset() #stochastic setting
            exploreIdagent = ExploreIDAgent(env, horizon = H0, Need = N, Gamma = Gamma)
        
            succ, explore_reward = exploreIdagent.run_episode_with_initial_state(initial_state)
            if succ is True:
                countSucc += 1
            #print("succ = {}".format(succ))
            print("m = {}, succ = {}".format(m, succ))
            end = time.time()
            print("Time = ", end - start, " seconds")
            
            print(exploreIdagent.N_sa.min(), exploreIdagent.N_sa.max())
            
    # seq = constants.seq
    # assert(len(seq) == K)
    # start = time.time()
    # for k in range(K):
        # m = seq[k]
        # env = env_list[m]
        # initial_state = env.reset()
        # exploreIdagent = ExploreIDAgent(env, horizon = H0, Need = N, Gamma = Gamma)
        # isSucc, explore_reward = exploreIdagent.run_episode_with_initial_state(initial_state)
        # if isSucc is True:
            # countSucc += 1
        # end = time.time()
        # print("k = {}, succ = {}, Time = {} seconds".format(k, isSucc, end - start))
        
        