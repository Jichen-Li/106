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

    def sample(self, nSamples):
        self.samples = [self.state]
        self.acceptanceRate = 0.4
        for i in range(nSamples):
            nextState = np.random.normal(loc=self.state, scale=self.stepSize)
            logAcceptanceProb = self.logTarget(nextState) - self.logTarget(self.state)
            if np.log(np.random.uniform()) < logAcceptanceProb:
                self.state = nextState
                self.samples.append(nextState)
                self.acceptanceRate += 1.0 / len(self.samples)
        return self
    
    def adapt(self, blockLengths):
        for i, blockLength in enumerate(blockLengths):
            startIdx = i * blockLength
            endIdx = startIdx + blockLength
            blockSamples = self.samples[startIdx:endIdx]
            blockAcceptanceRate = np.mean([self.logTarget(s) for s in blockSamples])
            if blockAcceptanceRate > 0.234:
                self.stepSize *= 1.1
            else:
                self.stepSize /= 1.1
        return self
    
    def summary(self):
        return {
            'mean': np.mean(self.samples),
            'sd': np.std(self.samples),
            'se': np.std(self.samples) / np.sqrt(len(self.samples)),
            'c025': np.percentile(self.samples, 2.5),
            'c975': np.percentile(self.samples, 97.5),
            'acceptanceRate': self.acceptanceRate
        }