import numpy as np

from gym import Env, spaces
from gym.utils import seeding
import random

def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()


class DiscreteEnv(Env):

    """
    Has the following members
    - nS: number of states
    - nA: number of actions
    - P: transitions (*)
    - isd: initial state distribution (**)

    (*) dictionary dict of dicts of lists, where
      P[s][a] == [(probability, nextstate, reward, done), ...]
    (**) list or array of length nS


    """
    def __init__(self, nS, nA, P, isd):
        self.P = P
        self.isd = isd
        self.lastaction = None # for rendering
        self.nS = nS
        self.nA = nA
        self.episode = 0 # number of episodes elapsed

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self.seed()
        self.s = categorical_sample(self.isd, self.np_random)
        self.ls = None # last state
        self.lastaction=None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def exploreReset(self):
        state = None
        if(self.s == 14):
            state = random.choice([12, 13])
        elif(self.s == 7):
            state = random.choice([6, 8])
        elif(self.s == 0):
            state = random.choice([1, 2])
        else:
            state = random.choice([36, 37,38])
        self.s = state
        
        #self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction = None
        return self.s
    
    def rewardsReset(self):
        self.s = categorical_sample(self.isd, self.np_random)
        while self.s in [0, 1, 3, 4, 6, 8]:
            self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction = None
        return self.s
    
    def fixedGoalReset(self):
        self.s = categorical_sample(self.isd, self.np_random)
        while self.s in [4, 8, 6, 1, 3]:
            self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction = None
        return self.s
    
    def centerReset(self):
        # state 48 = both available
        # state 50 = hare available
        # state 49 = rabbit available
        #self.s = categorical_sample(self.isd, self.np_random)
        
        if self.episode % 3 == 0:
            self.s = 48
        elif self.episode % 2 == 0:
            self.s = 50
        elif self.episode % 2 == 1:
            self.s = 49
        
        self.episode = self.episode + 1
        self.lastaction = None
        return self.s

    def step(self, a):
        self.ls = self.s
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d= transitions[i]
        self.s = s
        self.lastaction = a
        return (s, self.ls, r, d, {"prob" : p})


