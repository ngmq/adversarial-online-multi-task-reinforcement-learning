import argparse
import time
import numpy as np
import pandas as pd
import math
from copy import deepcopy
from rlberry.envs import FiniteMDP, GridWorld
from rlberry.agents import AgentWithSimplePolicy
from rlberry.agents.dynprog import ValueIterationAgent
from UCBVICHAgent import UCBVICHAgent
from ExploreIDAgent import ExploreIDAgent
from AOMTAgent import AOMTAgent
from RandomAgent import RandomAgent
import constants
import utils
from utils import Cluster

if __name__ == '__main__':    
    start = time.time()    
    sz = constants.sz
    env_list, Lambda, GammaAll, GammaCover = utils.create_environments(sz)
    M = len(env_list)
    A = env_list[0].A # 4 up down left right
    S = env_list[0].S    
    assert(M == constants.M and A == constants.A and S == sz * sz)
    K = constants.K
    p1 = constants.p1
    logTerm = utils.computeLogTerm(K, len(GammaCover), p1)
    N = utils.computeN(Lambda, logTerm)
    D = constants.D
    n = utils.computeNumberOfIntervals(N, logTerm)
    delta = utils.computeDelta(alpha = Lambda, Lambda = Lambda)
    
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
    
    print("N = {}, H0 = {}".format(N, H0))
    
    for runid in range(constants.nruns):
        ModelClusters = list()
        RegretClusters = list()
    
        data = np.zeros((K, 4)) # Four columns: optimal, AOMultiRL1, One-Ep and Random
        
        seq = utils.modelSequence()
        countSucc = 0
        for k in range(K):
            last_k = k 
            m = seq[k]
            env = env_list[m]
            print("Starting k = {}, m = {}".format(k, m))
            
            aomtAgent = AOMTAgent(env = env, H0 = H0, H1 = constants.H1, Need = N, Gamma = Gamma, failure_prob = constants.failure_prob, M = M, nepisodes = K)

            isSucc, explore_reward = aomtAgent.run_exploration()
            if isSucc is True:
                countSucc += 1
            # print("Exploration, isSucc = {}".format(isSucc))

            # Find cluster
            c = -1
            i = 0
            for cluster in ModelClusters:
                same = True
                for s in range(S):
                    for a in range(A):
                        if utils.isNotSameCluster(cluster.P_hat[s, a, :], aomtAgent.exploreAgent.P_hat[s, a, :], S, delta):
                            same = False
                            break
                    if not same:
                        break
                
                if same:
                    c = i
                    break
                i += 1
            
            if c == -1:
                print("new cluster!".format(m))
                cluster = Cluster(S, A)
                
                cluster.N_sa = deepcopy(aomtAgent.exploreAgent.N_sa)
                cluster.N_sas = deepcopy(aomtAgent.exploreAgent.N_sas)
                cluster.P_hat = deepcopy(aomtAgent.exploreAgent.P_hat)
                        
                ModelClusters.append(cluster)
                
                regretcluster = Cluster(S, A) # not combine
                # regretcluster = deepcopy(cluster) # combine
                RegretClusters.append(regretcluster)
                
                c = i
            else:
                print("exising cluster = {}".format(c))            
                for s in range(S):
                    for a in range(A):
                        for sprime in range(S): # change to neighbors
                        #for sprime in env.valid_neighbors[s]:
                            if aomtAgent.exploreAgent.N_sas[s, a, sprime] > 0:                            
                                ModelClusters[c].update(s, a, sprime, aomtAgent.exploreAgent.N_sas[s, a, sprime])
                                # combine
                                # RegretClusters[c].update(s, a, sprime, aomtAgent.exploreAgent.N_sas[s, a, sprime])
        
            # reference copy is intended
            aomtAgent.ucbviAgent.N_sa = RegretClusters[c].N_sa
            aomtAgent.ucbviAgent.N_sas = RegretClusters[c].N_sas
            aomtAgent.ucbviAgent.P_hat = RegretClusters[c].P_hat
            aomtAgent.run_optimization()      
            
            # now test the policy that aomtAgent learned
            aomtAgent.ucbviAgent.compute_policy()
            
            # one-episode UCBVI
            current_state = aomtAgent.exploreAgent.env.state
            oneEpUCBVI = UCBVICHAgent(env, horizon = constants.H1, nepisodes = 1, M = 1, failure_prob = constants.failure_prob / 3)
            oneEpUCBVI.run_episode(initial_state = current_state)
            oneEpUCBVI.compute_policy()
            
            # random agent
            randomAgent = RandomAgent(env, H1 = constants.H1)     

            # optimal agent
            optimalAgent = ValueIterationAgent(env, horizon=constants.H1, gamma = 1.0) # gamma is needed to make Q and V accurate
            optimalAgent.fit()
            
            # Test four agents
            
            c_learn_reward = aomtAgent.ucbviAgent.test_episode()
            c_oneep_reward = oneEpUCBVI.test_episode()
            c_random_reward = randomAgent.run_episode()   

            c_optimal_reward = 0
            state = optimalAgent.env.reset()
            for h in range(constants.H1):
                    action = optimalAgent.Q[h, state, :].argmax()
                    next_state, reward, done, _ = optimalAgent.env.step(action)

                    state = next_state
                    c_optimal_reward += reward         

            data[k, 0] = c_optimal_reward
            data[k, 1] = c_learn_reward
            data[k, 2] = c_oneep_reward
            data[k, 3] = c_random_reward
        
            # np.save("Data/AOMultiRL1/Run{}Ep{}.npy".format(sz, sz, runid, k), data[k])
        
            #print("Rewards: episode = {}, c_learn_reward = {}, c_oneep_reward = {}, c_random_reward = {}, c_optimal_reward = {}".format(k, c_learn_reward, c_oneep_reward, c_random_reward, c_optimal_reward))
        
            end = time.time()
            print("Elapsed Time = ", end - start, " seconds")
        
        print("Out of {} episodes, {} explorations were successful".format(K, countSucc))
        np.save("Data/AOMultiRL1/Run{}.npy".format(runid), data)
        end = time.time()
        print("One Run, Elapsed Time = ", end - start, " seconds")
    
    end = time.time()
    print("Final Time = ", end - start, " seconds")
    
    

    
    