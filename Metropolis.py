import numpy as np

class Metropolis:

    def __init__(self, logTarget, initialState, stepSize=0.1):
        self.logTarget = logTarget
        self.state = initialState
        self.stepSize = stepSize
        self.samples = []
        self.acceptanceRate = 0.4

    def step(self):
        nextState = np.random.normal(loc=self.state, scale=self.stepSize)
        logAcceptanceProb = self.logTarget(nextState) - self.logTarget(self.state)
        if np.log(np.random.uniform()) < logAcceptanceProb:
            self.state = nextState
            self.samples.append(nextState)
            self.acceptanceRate += 1.0 / len(self.samples)