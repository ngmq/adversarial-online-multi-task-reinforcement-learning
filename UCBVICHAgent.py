import time
import numpy as np
import pandas as pd
import math
from rlberry.envs import FiniteMDP, GridWorld
from rlberry.agents import AgentWithSimplePolicy
from rlberry.agents.dynprog import ValueIterationAgent
from rlberry.agents.dynprog.utils import backward_induction_in_place
from RandomAgent import RandomAgent
from utils import L1
# import logging

import gym.spaces as spaces
# from rlberry.exploration_tools.discrete_counter import DiscreteCounter
from rlberry.agents.dynprog.utils import backward_induction # for computing the optimal Q and V in finite-horizon MDPs

# logger = logging.getLogger(__name__)


class UCBVICHAgent(AgentWithSimplePolicy):
    """
    UCBVI [1]_ with custom exploration bonus.

    Notes
    -----
    The recommended policy after all the episodes is computed without
    exploration bonuses.

    Parameters
    ----------
    env : gym.Env
        Environment with discrete states and actions.
    horizon : int
        Horizon of the objective function.
    bonus_scale_factor : double, default: 1.0
        Constant by which to multiply the exploration bonus, controls
        the level of exploration.
    bonus_type : {"simplified_bernstein"}
        Type of exploration bonus. Currently, only "simplified_bernstein"
        is implemented. If `reward_free` is true, this parameter is ignored
        and the algorithm uses 1/n bonuses.    

    References
    ----------
    .. [1] Azar et al., 2017
        Minimax Regret Bounds for Reinforcement Learning
        https://arxiv.org/abs/1703.05449
    """
    name = "UCBVICH"

    def __init__(self,
                 env,                 
                 horizon=100,
                 bonus_scale_factor=1.0,
                 nepisodes=1,
                 failure_prob=0.3,
                 M = 1,
                 copy_env = True,
                 **kwargs):
        # init base class
        AgentWithSimplePolicy.__init__(self, env, copy_env = copy_env, **kwargs)
        
        if math.isclose(failure_prob, 0.0):
            assert "Failure probability must be greater than 0"
            
        self.S = self.env.observation_space.n
        self.A = self.env.action_space.n
        self.H = horizon
        self.bonus_scale_factor = bonus_scale_factor
        self.K = nepisodes
        self.failure_prob = failure_prob
        self.T = self.K * self.H
        self.M = M
        self.L = np.log(5 * self.S * self.A * self.T * self.M / self.failure_prob)
        self.HL = 2 * self.H * self.L   # the constant factor is 7 for Hoeffding     

        # check environment
        assert isinstance(self.env.observation_space, spaces.Discrete)
        assert isinstance(self.env.action_space, spaces.Discrete)

        # other checks        
        if self.H is None:
            assert "No horizon is given."
            

        # maximum value
        r_range = self.env.reward_range[1] - self.env.reward_range[0]
        if r_range == np.inf or r_range == 0.0:
            logger.warning("{}: Reward range is  zero or infinity. ".format(self.name)
                           + "Setting it to 1.")
            r_range = 1.0

        self.v_max = np.zeros(self.H)
        self.v_max[-1] = r_range
        for hh in reversed(range(self.H - 1)):
            self.v_max[hh] = r_range + self.v_max[hh + 1]

        # initialize
        self.reset()

    def resetEnv(self, env, initial_state):
        assert self.S == env.observation_space.n
        assert self.A == env.action_space.n        
        self.env = env
        self.env.setState(initial_state)
        
    def reset(self, **kwargs):
        shape_sa = (self.S, self.A)
        shape_sas = (self.S, self.A, self.S)

        # (s, a) visit counter
        self.N_sa = np.zeros(shape_sa)
        self.N_sas = np.zeros(shape_sas)
        # (s, a) bonus
        self.B_sa = np.ones(shape_sa)

        # MDP estimator
        self.R_hat = np.zeros(shape_sa)
        self.P_hat = np.zeros(shape_sas) # not the uniform distribution np.ones(shape_sas) * 1.0 / S

        # Value functions
        self.V = np.zeros((self.H+1, self.S))
        self.Q = np.zeros((self.H, self.S, self.A))
        # for testing policy
        self.V_policy = np.zeros((self.H, self.S))
        self.Q_policy = np.zeros((self.H, self.S, self.A))        

        # ep counter
        self.episode = 0

        # useful object to compute total number of visited states & entropy of visited states
        #self.counter = DiscreteCounter(self.env.observation_space,
        #                               self.env.action_space)

    def policy(self, observation):
        pass

    def _get_action(self, state, h=0):
        """ Sampling policy. """        
        assert self.Q is not None
        #return self.Q[h, state, :].argmax()
        
        Qmax = self.Q[h, state, :].max()
        amaxs = []
        for a in range(self.A):
            if math.isclose(self.Q[h, state, a], Qmax):
                amaxs.append(a)
                
        return np.random.choice(amaxs)

    def _compute_bonus(self, n):       
        assert n > 0
        # bonus = self.HL * np.sqrt(1.0 / n) # Hoeffding        
        bonus = self.H * np.sqrt(self.HL * 1.0 / n) + self.HL / (3.0 * n) # simplified_bernstein
        return bonus
        
    def _ucb_q_values(self):
        # compute bonuses and value iteration
        for s in range(self.S):
            self.V[self.H, s] = 0.0
            for a in range(self.A):
                if self.N_sa[s, a] > 0:                
                    self.B_sa[s, a] = self._compute_bonus(self.N_sa[s, a])
                    # if a == 0:
                        # print("s = {}, a = {}, n = {}, bonus = {}".format(s, a, self.N_sa[s, a], self.B_sa[s, a]))
                else:                                        
                    self.B_sa[s, a] = self.H
                        
            
        prev_Q = self.Q
        for h in range(self.H-1, -1, -1):
            # h = H-1 to 0
            for s in range(self.S):
                for a in range(self.A):
                    self.Q[h, s, a] = self.env.R[s, a] + self.B_sa[s, a]
                    tmp = 0.0
                    for sprime in range(self.S):
                        tmp += self.P_hat[s, a, sprime] * self.V[h+1, sprime]
                        
                    self.Q[h, s, a] += tmp
                    self.Q[h, s, a] = min(self.Q[h, s, a], self.H)
                    #self.Q[h, s, a] = min(self.Q[h, s, a], prev_Q[h, s, a])
                    
                    # if not math.isclose(self.Q[h, s, a], 0.0):
                        # print("time step = {}, state = {}, a = {}. Not".format(h, s, a))
                    
                self.V[h, s] = self.Q[h, s, :].max()

    def _update(self, state, action, next_state):
        self.N_sa[state, action] += 1
        self.N_sas[state, action, next_state] += 1

        nn = self.N_sa[state, action]        
        prev_p = self.P_hat[state, action, :]
        
        self.P_hat[state, action, :] = (1.0 - 1.0 / nn) * prev_p
        self.P_hat[state, action, next_state] += 1.0 / nn 
        
    def run_episode(self, initial_state = None):
        # compute Q and V functions
        self._ucb_q_values()
        
        # interact for H steps
        episode_rewards = 0
        if initial_state is not None:            
            state = initial_state
            #self.env.setState(state)
            self.env.state = state
        else:
            self.env.reset()
            
        state = self.env.state
        
        for h in range(self.H):
            action = self._get_action(state, h)
            # logger.info("state = {}, action is {}".format(state, action))
            next_state, reward, done, _ = self.env.step(action)
            episode_rewards += reward  # used for logging only

            self._update(state, action, next_state)
            
            state = next_state

        # update info
        self.episode += 1
        
        # return sum of rewards collected in the episode
        # logger.info(episode_rewards)
        
        return episode_rewards
        
    def fit(self):        
        count = 0
        while count < self.K:
            initial_state = self.env.reset() #stochastic setting
            # initial_state = 3 # adversarial setting
            # logger.info(type(initial_state)) # <class 'int'>
            self.run_episode(initial_state)
            count += 1
            
    def compute_policy(self):
        backward_induction_in_place(
                Q = self.Q_policy,
                V = self.V_policy,
                R = self.env.R,
                P = self.P_hat,
                horizon = self.H,
                vmax = self.H)
                
    def test_episode(self, initial_state = None):
        # interact for H steps
        episode_rewards = 0
        if initial_state is not None:            
            state = initial_state
            self.env.setState(state)
        else:
            self.env.reset()
            
        state = self.env.state
        
        for h in range(self.H):
            # action = self._get_action(state, h)
            action = self.Q_policy[h, state, :].argmax()
            # logger.info("state = {}, action is {}".format(state, action))
            next_state, reward, done, _ = self.env.step(action)
            episode_rewards += reward  # used for logging only

            # !!! no update while testing 
            # self._update(state, action, next_state)
            
            state = next_state
        
        # return sum of rewards collected in the episode
        # logger.info(episode_rewards)
        
        return episode_rewards
        

if __name__ == '__main__':
    np.random.seed(0)
    
    S = 9
    A = 4
    H = 20
    K = 100000

    R = np.random.uniform(0, 1, (S, A))
    P = np.random.uniform(0, 1, (S, A, S))
    initial_state_distr = 0  # np.ones(S)/S
    for ss in range(S):
        for aa in range(A):
            P[ss, aa, :] /= P[ss, aa, :].sum()

    env = FiniteMDP(R, P, initial_state_distribution=initial_state_distr)
    optimalQ, optimalV = backward_induction(env.R, env.P, horizon = H)
    print("Optimal Value is {}".format(optimalV[0, initial_state_distr]))
    
    optimalAgent = ValueIterationAgent(env, horizon=H)
    optimalAgent.reseed(0)
    info = optimalAgent.fit()
    
    # env.reseed(233) # Be careful with the re-seed in AgentWithSimplePolicy.__init__(self, env, **kwargs)
     
    ucbvichagent = UCBVICHAgent(env, horizon = H, nepisodes = K)
    ucbvichagent.reseed(233) # be careful with the re-seed in AgentWithSimplePolicy class
    # ucbvichagent.fit()
    start = time.time()
    for k in range(K):
        train_ep_reward = ucbvichagent.run_episode()
        end = time.time()
        if k % 1000 == 0:
            print("k = {}".format(k), "; Time = ", end - start, " seconds")
        
    end = time.time()
    print("Final Time = ", end - start, " seconds")
    
    print("After training")
    for s in range(S):
        for a in range(A):
            print("(s, a) = ({}, {}), visits = {}, bonus = {}".format(s, a, ucbvichagent.N_sa[s, a], ucbvichagent.B_sa[s, a]))
            print("L1 error = {}".format(L1(ucbvichagent.P_hat[s, a], env.P[s, a], S)))
            
    s = 5
    a = 2
    for h in range(H):
        print("Optimistic Q[{}, s, a] = {}".format(h, ucbvichagent.Q[h, s, a]))
    
    
    # ucbvichagent.compute_policy()
    # s = 0
    # for h in range(H):
        # print("h = {}".format(h), ucbvichagent.Q_policy[h, s, 0], optimalAgent.Q[h, s, 0])
            
            
    for _ in range(10):
        ep_reward = ucbvichagent.run_episode()
        # ep_reward = ucbvichagent.test_episode()
        
        optimal_ep_reward = 0
        state = env.reset()
        for h in range(H):
            # action = optimalAgent.policy(state)
            action = optimalAgent.Q[h, state, :].argmax()            
            next_s, reward, done, _ = env.step(action)            
            state = next_s
            
            optimal_ep_reward += reward
        
        randomAgent = RandomAgent(env = env, H1 = H)
        state = env.reset()
        random_reward = randomAgent.run_episode_with_initial_state(state)
        print("random_reward = {}, train_ep_reward = {}, optimal_ep_reward = {}".format(random_reward, ep_reward, optimal_ep_reward))