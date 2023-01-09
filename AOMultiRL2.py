import pickle
import argparse
import time
import numpy as np
import pandas as pd
import math
from copy import deepcopy
from rlberry.envs import FiniteMDP, GridWorld
from rlberry.agents import AgentWithSimplePolicy
from UCBVICHAgent import UCBVICHAgent
from ExploreIDAgent import ExploreIDAgent
from AOMTAgent import AOMTAgent
from RandomAgent import RandomAgent
import constants
import utils
from utils import Cluster

import gym.spaces as spaces

# logger = logging.getLogger(__name__)

def run_Stage1(env_list, Lambda, runid):
    M = len(env_list)
    A = env_list[0].A # 4 up down left right
    S = env_list[0].S    
    # assert(M == constants.M and A == constants.A and S == constants.S)
    K = constants.K
    p1 = constants.p1
    ########### Stage 1 ##############
    logTerm = utils.computeLogTerm(K, S * A, p1)
    N = utils.computeN(Lambda, logTerm)
    D = constants.D
    n = utils.computeNumberOfIntervals(N, logTerm)
    delta = utils.computeDelta(alpha = Lambda, Lambda = Lambda)
    
    Gamma = dict()
        
    for s in range(S):
        Gamma[s] = set() 
        for a in range(A):
            Gamma[s].add(a)
                
    H0 = int(2 * D * S * A * n)
    #H0 = int(D * S * A * N)
    
    print("Stage 1: N = {}, H0 = {}".format(N, H0))
    
    ModelClusters = list()
    RegretClusters = list()
    
    start = time.time() 
    for m in range(M):
        env = env_list[m]
        aomtAgent = AOMTAgent(env = env, H0 = H0, H1 = constants.H1, Need = N, Gamma = Gamma, failure_prob = constants.failure_prob, M = M, nepisodes = K)
        aomtAgent.reseed(runid)
        
        isSucc, explore_reward = aomtAgent.run_exploration()
        
        # if isSucc is True:
            # countSucc += 1
        print("Stage1 Exploration, isSucc = {}".format(isSucc))
        
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
                            
        learn_reward = aomtAgent.run_optimization()
        
        end = time.time()
        print("Time = ", end - start, " seconds")
    
    end = time.time()
    print("Final Stage 1 Time = ", end - start, " seconds")
    print("Number of clusters = {}".format(len(ModelClusters)))
    
    with open("Data/AOMultiRL2/ModelClustersRun{}.dat".format(runid), "wb") as f:
        pickle.dump(ModelClusters, f)
        
    with open("Data/AOMultiRL2/RegretClustersRun{}.dat".format(runid), "wb") as f:
        pickle.dump(RegretClusters, f)
                            
    return ModelClusters, RegretClusters
    
def discoverGamma(ModelClusters, RegretClusters, Lambda, S, A):
    C = len(ModelClusters)
    GammaHat = set()
    
    for i in range(C):
        for j in range(i + 1, C):
            # print("=== i = {}, j = {} ===".format(i, j))
            found = False
            for s in range(S-1, -1, -1):
                for a in range(A):
                    tmp = utils.L1(ModelClusters[i].P_hat[s, a, :], ModelClusters[j].P_hat[s, a, :], S)
                    if  tmp > 3 * Lambda / 4:
                        # print("(s, a) = ({}, {}), empirical L1 = {}".format(s, a, tmp))
                        GammaHat.add((s,a))
                        found = True
                        break
                if found:
                    break
                    
    #print("GammaHat is {}".format(GammaHat))
    return GammaHat
                            
if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Run ID')
    # parser.add_argument("runid")
    # args = parser.parse_args()
    # print(args.runid)
    
    """
    Experiment 2: do not use GammaCover. Discover it instead.
    """
    
    start = time.time()    
    sz = constants.sz
    env_list, Lambda, GammaAll, GammaCover = utils.create_environments(sz)
    M = len(env_list)
    A = env_list[0].A # 4 up down left right
    S = env_list[0].S 
    
    for runid in range(constants.nruns):
        ModelClusters, RegretClusters = run_Stage1(env_list, Lambda, runid) # run once 
    
        # ModelClusters = None
        # with open("Results/4x4/Experiment2/ModelClustersRun{}.dat".format(runid), "rb") as f:        
            # ModelClusters = pickle.load(f)
            
        # assert ModelClusters is not None
        # assert len(ModelClusters) == len(env_list)
    
        # RegretClusters = None
        # with open("Results/4x4/Experiment2/RegretClustersRun{}.dat".format(runid), "rb") as f:
            # RegretClusters = pickle.load(f)
        
        assert ModelClusters is not None
        assert RegretClusters is not None
        assert len(RegretClusters) == len(ModelClusters)
    
        GammaHat = discoverGamma(ModelClusters, RegretClusters, Lambda, S, A)
        print("Runid = {}, GammaHat = {}".format(runid, GammaHat))
    
        ############## Stage 2
        ## Recompute logTerm, N, n, delta and H0
        alpha = Lambda / 2
        Gamma = dict()
        for s, a in GammaHat:
            if s not in Gamma:
                Gamma[s] = set()
            Gamma[s].add(a)    
        
        K = constants.K
        p1 = constants.p1
        logTerm = utils.computeLogTerm(K, len(GammaHat), p1)
        N = utils.computeN(Lambda, logTerm)
        D = constants.D
        n = utils.computeNumberOfIntervals(N, logTerm)
        delta = utils.computeDelta(alpha = alpha, Lambda = Lambda)
        H0 = int(2 * D * len(GammaHat) * n)
        print("N = {}, H0 = {}".format(N, H0))
    
        seq = utils.modelSequence()
        countSucc = 0
        data = np.zeros((K, 1))
    
        for k in range(K):
            m = seq[k]
            env = env_list[m]
            print("Starting k = {}, m = {}".format(k, m))
            
            aomtAgent = AOMTAgent(env = env, H0 = H0, H1 = constants.H1, Need = N, Gamma = Gamma, failure_prob = constants.failure_prob, M = M, nepisodes = K)
            # aomtAgent.reseed(runid)
            
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
                RegretClusters.append(regretcluster)
                
                c = i
            else:
                print("exising cluster = {}".format(c))            
                for s in range(S):
                    for a in range(A):
                        for sprime in range(S):
                            if aomtAgent.exploreAgent.N_sas[s, a, sprime] > 0:                            
                                ModelClusters[c].update(s, a, sprime, aomtAgent.exploreAgent.N_sas[s, a, sprime])
            
            # reference copy is intended
            aomtAgent.ucbviAgent.N_sa = RegretClusters[c].N_sa
            aomtAgent.ucbviAgent.N_sas = RegretClusters[c].N_sas
            aomtAgent.ucbviAgent.P_hat = RegretClusters[c].P_hat
            aomtAgent.run_optimization()        
            
            aomtAgent.ucbviAgent.compute_policy()
            
            c_learn_reward = aomtAgent.ucbviAgent.test_episode()             
                
            data[k, 0] = c_learn_reward            
            
            print("Rewards: episode = {}, c_learn_reward = {}".format(k, c_learn_reward))
            end = time.time()
            print("Time = ", end - start, " seconds")
            
        print("Out of {} episodes, {} were successful".format(K, countSucc))
        np.save("Data/AOMultiRL2/Run{}.npy".format(runid), data)
        end = time.time()
        print("One Run, Elapsed Time = ", end - start, " seconds")
        
    end = time.time()
    print("Final Time = ", end - start, " seconds")
    
    

    
    