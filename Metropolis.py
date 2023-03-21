import numpy as np

class Metropolis:

    def __init__(self, logTarget, initialState, stepSize=0.1):
        self.logTarget = logTarget
        self.state = initialState
        self.stepSize = stepSize
        self.samples = []
        self.acceptanceRate = 0.4

    def _accept(self, proposal):
        log_ratio = self.log_target(proposal) - self.log_target(self.current_state)
        acceptance_probability = np.exp(np.minimum(log_ratio, 0))
        if np.random.uniform() < acceptance_probability:
            self.current_state = proposal
            self.accepted_proposals += 1
            return True
        else:
            return False

    def step(self):
        nextState = np.random.normal(loc=self.state, scale=self.stepSize)
        logAcceptanceProb = self.logTarget(nextState) - self.logTarget(self.state)
        if np.log(np.random.uniform()) < logAcceptanceProb:
            self.state = nextState
            self.samples.append(nextState)
            self.acceptanceRate += 1.0 / len(self.samples)