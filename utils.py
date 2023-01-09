import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math
from rlberry.envs import FiniteMDP, GridWorld
from rlberry.agents.dynprog import ValueIterationAgent
import constants

def L1(dist1, dist2, S):
    ret = 0
    for s in range(S):
        ret += abs(dist1[s] - dist2[s])
    return ret
    
def isNotSameCluster(dist1, dist2, S, delta):
    return L1(dist1, dist2, S) > delta
        
class Cluster:
    name = "Cluster"

    def __init__(self, S, A):
        self.S = S
        self.A = A        
        
        shape_sa = (self.S, self.A)
        shape_sas = (self.S, self.A, self.S)

        # (s, a) visit counter
        self.N_sa = np.zeros(shape_sa)
        self.N_sas = np.zeros(shape_sas)
        
        self.P_hat = np.zeros(shape_sas) # start from all zeros, not uniform distribution
    
    def update(self, state, action, next_state, nadd):
        assert nadd >= 1
        self.N_sa[state, action] += nadd

        nn = self.N_sa[state, action]        
        prev_p = self.P_hat[state, action, :]
        
        self.P_hat[state, action, :] = (1.0 - 1.0 * nadd / nn) * prev_p
        self.P_hat[state, action, next_state] += 1.0 * nadd / nn      

def blockBottomRight(sz, env):
    constant = 0.3
    
    s = env.coord2index[(sz-2, sz-1)]
    a = env.a_str2idx['down']
    env.P[s, a, env.coord2index[(sz-1, sz-1)]] = constant
    env.P[s, a, env.coord2index[(sz-2, sz-2)]] = (1.0 - constant) / 2
    env.P[s, a, env.coord2index[(sz-3, sz-1)]] = (1.0 - constant) / 2
    
    #print("s = {}, a = {}, p = {}".format(s, a, env.P[s, a]))
    
    s = env.coord2index[(sz-1, sz-2)]
    a = env.a_str2idx['right']
    env.P[s, a, env.coord2index[(sz-1, sz-1)]] = constant
    env.P[s, a, env.coord2index[(sz-2, sz-2)]] = (1.0 - constant) / 2
    env.P[s, a, env.coord2index[(sz-1, sz-3)]] = (1.0 - constant) / 2
    
    #print("s = {}, a = {}, p = {}".format(s, a, env.P[s, a]))
    
def blockTopRight(sz, env):
    constant = 0.2
    
    s = env.coord2index[(0, sz-2)]
    a = env.a_str2idx['right']
    env.P[s, a, env.coord2index[(0, sz-1)]] = constant
    env.P[s, a, env.coord2index[(1, sz-2)]] = (1.0 - constant) / 2
    env.P[s, a, env.coord2index[(0, sz-3)]] = (1.0 - constant) / 2
    
    #print("s = {}, a = {}, p = {}".format(s, a, env.P[s, a]))
    
    s = env.coord2index[(1, sz-1)]
    a = env.a_str2idx['up']
    env.P[s, a, env.coord2index[(0, sz-1)]] = constant
    env.P[s, a, env.coord2index[(2, sz-1)]] = (1.0 - constant) / 2
    env.P[s, a, env.coord2index[(1, sz-2)]] = (1.0 - constant) / 2   
    
    #print("s = {}, a = {}, p = {}".format(s, a, env.P[s, a]))
    
def blockBottomLeft(sz, env):
    constant = 0.2
    
    s = env.coord2index[(sz-2, 0)]
    a = env.a_str2idx['down']
    env.P[s, a, env.coord2index[(sz-1, 0)]] = constant
    env.P[s, a, env.coord2index[(sz-3, 0)]] = (1.0 - constant) / 2
    env.P[s, a, env.coord2index[(sz-2, 1)]] = (1.0 - constant) / 2
    
    #print("s = {}, a = {}, p = {}".format(s, a, env.P[s, a]))
    
    
    s = env.coord2index[(sz-1, 1)]
    a = env.a_str2idx['left']
    env.P[s, a, env.coord2index[(sz-1, 0)]] = constant
    env.P[s, a, env.coord2index[(sz-2, 1)]] = (1.0 - constant) / 2
    env.P[s, a, env.coord2index[(sz-1, 2)]] = (1.0 - constant) / 2
    
    #print("s = {}, a = {}, p = {}".format(s, a, env.P[s, a]))
    
def blockTopLeft(sz, env):    
    constant = 0.2
    
    s = env.coord2index[(1, 0)]
    a = env.a_str2idx['up']
    env.P[s, a, env.coord2index[(0, 0)]] = constant
    env.P[s, a, env.coord2index[(2, 0)]] = (1.0 - constant) / 2
    env.P[s, a, env.coord2index[(1, 1)]] = (1.0 - constant) / 2
    
    #print("s = {}, a = {}, p = {}".format(s, a, env.P[s, a]))
    
    s = env.coord2index[(0, 1)]
    a = env.a_str2idx['left']
    env.P[s, a, env.coord2index[(0, 0)]] = constant
    env.P[s, a, env.coord2index[(0, 2)]] = (1.0 - constant) / 2
    env.P[s, a, env.coord2index[(1, 1)]] = (1.0 - constant) / 2
    
    #print("s = {}, a = {}, p = {}".format(s, a, env.P[s, a]))
    
def makes_corners_hard_to_stay(sz, env):
    constant = 0.3
    s = env.coord2index[(0, 0)]
    a = env.a_str2idx['left']    
    env.P[s, a, env.coord2index[(0, 1)]] = constant
    env.P[s, a, s] = 1.0 - constant
    
    a = env.a_str2idx['up']    
    env.P[s, a, env.coord2index[(1, 0)]] = constant
    env.P[s, a, s] = 1.0 - constant
    
    s = env.coord2index[(0, sz-1)]
    a = env.a_str2idx['right']    
    env.P[s, a, env.coord2index[(0, sz-2)]] = constant
    env.P[s, a, s] = 1.0 - constant
    
    a = env.a_str2idx['up']    
    env.P[s, a, env.coord2index[(1, sz-1)]] = constant
    env.P[s, a, s] = 1.0 - constant
    
    s = env.coord2index[(sz-1, sz-1)]
    a = env.a_str2idx['right']    
    env.P[s, a, env.coord2index[(sz-1, sz-2)]] = constant
    env.P[s, a, s] = 1.0 - constant
    
    a = env.a_str2idx['down']    
    env.P[s, a, env.coord2index[(sz-2, sz-1)]] = constant
    env.P[s, a, s] = 1.0 - constant
    
    s = env.coord2index[(sz-1, 0)]
    a = env.a_str2idx['left']    
    env.P[s, a, env.coord2index[(sz-1, 1)]] = constant
    env.P[s, a, s] = 1.0 - constant
    
    a = env.a_str2idx['down']    
    env.P[s, a, env.coord2index[(sz-2, 0)]] = constant
    env.P[s, a, s] = 1.0 - constant
    
    
def create_environments(sz):
    # sz = 4 # try 3x3 to test speed
    S = sz * sz 
    A = 4
    M = 4

    env_list = list()
    
    for m in range(M):
        # R = np.random.uniform(0, 1, (S, A))
        # for s in range(S):
            # R[s, 0] = 0
            # R[s, 1] = 0.5
            
        # P = np.random.uniform(0, 1, (S, A, S))
        # initial_state_distr = 0  # np.ones(S)/S
        # for ss in range(S):
            # for aa in range(A):
                # P[ss, aa, :] /= P[ss, aa, :].sum()

        env = GridWorld(nrows = sz, ncols = sz, start_coord = (1, 1), success_probability = 0.85, walls = None, reward_at = {(0, sz-1) : 1.0, (sz-1, 0) : 1.0, \
                                                                                                     (sz-1, sz-1) : 1.0, (0, 0) : 1.0})
        # env.reseed(233)
        # print(env.S)
        # print(env.A)
        # print("--- m = {} ---".format(m))
        # for s in range(env.S):
            # print("** s = {} **".format(s))
            # for a in range(env.A):
                # print("+ action {}".format(env.a_idx2str[a]))
                # for ns in range(env.S):
                    # if not math.isclose(env.P[s, a, ns], 0.0):
                        # print("to {} with prob {}".format(ns, env.P[s, a, ns]))
        
        makes_corners_hard_to_stay(sz, env)
        
        if m == 0:      
            blockBottomLeft(sz, env)
            blockBottomRight(sz, env)
            blockTopRight(sz, env)
        elif m == 1:
            blockTopLeft(sz, env)
            blockBottomRight(sz, env)
            blockBottomLeft(sz, env)
        elif m == 2:
            blockTopLeft(sz, env)
            blockBottomLeft(sz, env)
            blockTopRight(sz, env)
        elif m == 3:
            blockTopLeft(sz, env)
            blockTopRight(sz, env)
            blockBottomRight(sz, env)           
            
        env_list.append(env)
        
    Lambda = 2.0
    
    for m1 in range(M):
        for m2 in range(m1 + 1, M):
            diff = 0
            for s in range(S):
                for a in range(A):
                    tmp = L1(env_list[m1].P[s, a, :], env_list[m2].P[s, a, :], S)
                    if tmp > diff:
                        diff = tmp
            if Lambda > diff:
                Lambda = diff
                
    Lambda -= 0.000001
    print("Lambda = {}".format(Lambda))
    
    for m1 in range(M):
        for m2 in range(m1 + 1, M):
            for s in range(S):
                for a in range(A):
                    tmp = L1(env_list[m1].P[s, a, :], env_list[m2].P[s, a, :], S)
                    if tmp >= Lambda/2:
                        print("m1 = {}, m2 = {}, (s, a) = ({}, {}), L1 = {}".format(m1, m2, s, a, tmp))
    
    GammaAll = {}
    GammaCover = set()
    for m1 in range(M):
        for m2 in range(m1 + 1, M):
            GammaAll[(m1, m2)] = list()
            added = False
            # print("m1 = {}, m2 = {}".format(m1, m2))
            for s in range(S):
                for a in range(A):
                    tmp = L1(env_list[m1].P[s, a, :], env_list[m2].P[s, a, :], S)
                    if (math.isclose(tmp, Lambda)) or tmp > Lambda:                        
                        GammaAll[(m1, m2)].append((s, a))
                        if not added:
                            GammaCover.add((s,a))
                            added = True
                        # print("({}, {})".format(s, a))
                        
    print(GammaCover)     
    return env_list, Lambda, GammaAll, GammaCover
    
def computeLogTerm(K, sz_set, p1):
    return np.log(K * sz_set / p1)
    
def computeN(Lambda, logTerm):
    return int(128.0 / (Lambda * Lambda) * (4 * np.log(2) + logTerm))
    
def computeNumberOfIntervals(N, logTerm):
    return int(2 * (N + logTerm) + 2 * np.sqrt(2 * N * logTerm + logTerm * logTerm))  # cannot be simplified further  
    
def computeDelta(alpha, Lambda):
    assert math.isclose(alpha, Lambda/2) or alpha > Lambda / 2
    return alpha - Lambda / 4

def visualization_exps(sz):
    K = constants.K
    nruns = constants.nruns
    l_df = np.zeros((4, nruns * K, 3))
    for agent in range(4):
        i = 0
        for runid in range(0, nruns):
            filename = "Data/AOMultiRL1/Run{}.npy".format(runid)
            run_data = np.load(filename)
            s = 0
            for k in range(K):
                s += run_data[k, agent]
                l_df[agent, i, 0] = runid
                l_df[agent, i, 1] = k
                l_df[agent, i, 2] = s / (k+1) #run_data[k, 1]
                i += 1

    # two-stage agent
    l_df_two_stage = np.zeros((nruns * K, 3))
    i = 0
    for runid in range(0, nruns):
        filename = "Data/AOMultiRL2/Run{}.npy".format(runid)
        run_data = np.load(filename)
        s = 0
        for k in range(K):
            s += run_data[k, 0]
            l_df_two_stage[i, 0] = runid
            l_df_two_stage[i, 1] = k
            l_df_two_stage[i, 2] = s / (k+1) #run_data[k, 1]
            i += 1
                
    df = [None] * 5
    # first four
    for agent in range(4):
        df[agent] = pd.DataFrame(l_df[agent], columns = ['runid', 'ep', 'reward'])
    df[4] = pd.DataFrame(l_df_two_stage, columns = ['runid', 'ep', 'reward'])

    df[2], df[3], df[4] = df[4], df[2], df[3]
    fig, ax = plt.subplots(figsize=(7, 5))
    for agent in range(5):
        sns.lineplot(data=df[agent], x="ep", y="reward", ci=None)
    ax.lines[0].set_linestyle('dashed')
    ax.lines[4].set_linestyle('dotted')
    ax.lines[2].set_linestyle('dashdot')
    ax.lines[3].set_linestyle((0, (5, 1)))
    #sns.despine()
    ax.legend(loc=(0.50, 0.15), labels=['Optimal non-stationary agent', 'AOMultiRL with $\Gamma$ given','AOMultiRL with $\hat{\Gamma}$ discovered', 'One-Episode UCBVI', 'Uniformly random agent'])
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average per-episode reward') 
    #plt.savefig("results.pdf", dpi=300)
    plt.show()
    
    
def modelSequence():
    np.random.seed(0)
    K = constants.K
    M = constants.M
    
    if K == 200:
        seq = list()
        for k in range(K):
            if (k >= 100 and k < 150) or (k >= K-20):
                m = M-1 # 3
            else:
                m = np.random.randint(M-1) # 0, 1, 2
            seq.append(m)
        return seq
    else:
        seq = list()
        for k in range(K):
            m = np.random.randint(M) # 0, 1, 2 and 3
            seq.append(m)
        return seq
        
    
if __name__ == '__main__':
    np.random.seed(0)

    visualization_exps(constants.sz)