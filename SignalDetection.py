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
        point = [self.fa_rate, self.hit_rate] # a vector of the data point's coordinate
        start = [0,0]
        end = [1,1]
        line1_x = [start[0],point[0]]
        line2_x = [point[0],end[0]] # x-axis characteristics of the two connecting lines
        line1_y = [start[1],point[1]]
        line2_y = [point[1],end[1]] # y-axis characteristics of the two connecting lines
        plt.plot(self.fa_rate, self.hit_rate, marker = 'o') # plot the data point
        plt.plot(line1_x,line1_y,'o')
        plt.plot(line2_x,line2_y,'o') # plot the lines
        plt.plot([0,1],'--') # Performance by chance
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.show()


    def plot_sdt(self):
        d = SignalDetection.d_prime(self)
        criterion = self.criterion()
        x = np.linspace(-6, 6, 100) # assume that standard deviation is 1
        plt.plot(x, norm.pdf(x, 0 , 1),label = 'noise')
        plt.plot(x, norm.pdf(x, -d , 1),label = 'signal')
        plt.axvline(x = criterion, color = 'black', label = 'criterion')
        plt.legend()
        plt.show()


### The code below is unrelated to hw3

    # def plot_roc(self):
    #     hit_list = [] # create a list containing hit counts across trials
    #     for i in range(len(self.hits)):
    #         hit_list.append(self.hit_rate[i])
    #     fa_list = []
    #     for j in range(len(self.FA)):
    #         fa_list.append(self.fa_rate[j])
    #     o1 = np.array([0,1])
    #     fa_list = np.sort(fa_list)
    #     hit_list = np.sort(hit_list)
    #     plt.figure(figsize=(6,6))
    #     plt.plot(o1, '--' , label = 'Optimal') # Optimal Performance
    #     plt.plot(fa_list , hit_list)
    #     plt.xlabel('False Alarm Rate')
    #     plt.ylabel('Hit Rate')
    #     plt.legend()
    #     plt.show()