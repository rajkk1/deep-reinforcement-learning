import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, eps_start=1, eps_discount=0.9999, eps_min=0.0000, alpha=0.1, gamma=1):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.eps = eps_start
        self.eps_discount = eps_discount
        self.eps_min = eps_min
        self.alpha = alpha
        self.gamma = 1

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if state in self.Q:
            probs = np.ones(self.nA)*self.eps/self.nA
            best_a = np.argmax(self.Q[state])
            probs[best_a] = 1-self.eps+self.eps/self.nA
            return np.random.choice(np.arange(self.nA), p=probs)
        else:
            return np.random.choice(self.nA)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        probs = np.ones(self.nA)*self.eps/self.nA
        best_a = np.argmax(self.Q[next_state])
        probs[best_a] = 1-self.eps+self.eps/self.nA
        mean_q = np.dot(self.Q[next_state], probs)
        self.Q[state][action] += self.alpha*(reward + self.gamma*mean_q - self.Q[state][action])
        self.eps = max(self.eps*self.eps_discount, self.eps_min)
