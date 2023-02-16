import numpy as np
from scipy.special import ndtri

class SignalDetection:

    def __init__(self, hits, misses, FA, CR):
        self.hits = hits
        self.misses = misses
        self.FA = FA
        self.CR = CR

    def hit_rate(self):
        self.hit_rate = self.hits/(self.hits + self.misses)
    def fa_rate(self):
        self.fa_rate = self.FA/(self.FA + self.CR)

    def d_prime(self):
        self.d_prime = ndtri(self.hit_rate) - ndtri(self.fa_rate)
        return self.d_prime
    
    def criterion (self):
        self.criterion = -0.5*(ndtri(self.hit_rate) + ndtri(self.fa_rate))
        return self.criterion

