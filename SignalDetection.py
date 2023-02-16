import numpy as np
from scipy.stats import norm
from scipy.special import ndtri
import matplotlib.pyplot as plt

class SignalDetection:

    def __init__(self, hits, misses, FA, CR):
        self.hits = hits
        self.misses = misses
        self.FA = FA
        self.CR = CR
        self.hit_rate = hits/(hits + misses)
        self.fa_rate = FA/(FA + CR)

    def d_prime(self):
        self.d_prime = ndtri(self.hit_rate) - ndtri(self.fa_rate)
        return self.d_prime
    
    def criterion (self):
        self.criterion = -0.5*(ndtri(self.hit_rate) + ndtri(self.fa_rate))
        return self.criterion

    def __add__ (self, other):
        return SignalDetection(self.hits + other.hits , self.misses + other.misses , self.FA + other.FA , self.CR + other.CR)

    def __mul__ (self, scalar):
        return SignalDetection(self.hits * scalar , self.misses * scalar , self.FA * scalar , self.CR * scalar)

    def plot_roc(self):
        o1 = np.array([0,1])
        plt.figure(figsize=(6,6))
        plt.plot(o1, '--' , label = 'Optimal')
        plt.plot(self.fa_rate , self.hit_rate)
        plt.xlabel('False Alarm Rate')
        plt.ylabel('Hit Rate')
        plt.legend()
        plt.show()

    def plot_sdt(self):
        c = SignalDetection.d_prime()/2
        x = np.linspace(-4*c, 4*c, 100) # assume that standard deviation is 1
        plt.plot(x, norm.pdf(x, c , 1))
        plt.plot(x, norm.pdf(x, -c , 1))
        plt.legend({'signal','noise'})
        plt.show()