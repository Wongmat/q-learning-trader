"""  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Template for implementing QLearner  (c) 2015 Tucker Balch  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Atlanta, Georgia 30332  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
All Rights Reserved  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Template code for CS 4646/7646  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
and other users of this template code are advised not to share it with others  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
or to make it available on publicly viewable websites including repositories  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
or edited.  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
We do grant permission to share solutions privately with non-students such  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
as potential employers. However, sharing with other current or future  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
GT honor code violation.  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
-----do not edit anything above this line---  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Student Name: Mathew Kai Yin Wong (replace with your name)
o

GT User ID: mwong83 (replace with your User ID)
GT ID: 903563087 (replace with your GT ID)
"""


import random as rand

import numpy as np


class QLearner(object):

    def __init__(self,
                 num_states=100,
                 num_actions=4,
                 alpha=0.2,
                 gamma=0.9,
                 rar=0.5,
                 radr=0.99,
                 dyna=0,
                 verbose=False):
        self.num_states = num_states
        self.num_actions = num_actions
        self.dyna = dyna
        self.gamma = gamma
        self.alpha = alpha
        self.rar = rar
        self.radr = radr
        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0
        self.q = np.random.uniform(
            low=0.01, high=0.1, size=(num_states, num_actions))
        self.t_c = np.full(
            (num_states, num_actions, num_states), 0.000000001)
        self.r = np.zeros((num_states, num_actions))

    def find_best_action(self, s):
        return np.argmax(self.q[s, :])

    def get_action(self, s):
        if self.rar > 0:
            rand_val = rand.uniform(0, 1)
            if rand_val <= self.rar:
                return rand.randint(0, self.num_actions-1)

        return self.find_best_action(s)

    def improved_estimate(self, s_prime, r):
        a_prime = self.find_best_action(s_prime)
        discounted_max = self.gamma * \
            self.q[s_prime, a_prime]
        return r + discounted_max

    def update_q(self, s, a, s_prime, r):
        old_reward = self.q[s, a]
        new_reward = (1 - self.alpha) * old_reward + \
            self.alpha * self.improved_estimate(s_prime, r)
        self.q[s, a] = new_reward

    def __set_custom_a(self, a):
        self.a = a

    def __set_custom_s(self, s):
        self.s = s

    def querysetstate(self, s):
        self.s = s
        self.a = self.get_action(s)
        if self.verbose:
            print(f"s = {self.s}, a = {self.a}")

        self.rar *= self.radr
        return self.a

    def query(self, s_prime, r):
        self.update_q(self.s, self.a, s_prime, r)
        if self.dyna > 0:
            self.run_dyna(s_prime, r)
        self.s = s_prime
        self.a = self.get_action(s_prime)
        if self.verbose:
            print(f"s = {self.s}, a = {self.a}, r={r}")

    def run_dyna(self, s_prime, r):
        def simulate():
            norm_tc = self.t_c / \
                self.t_c.sum(axis=2, keepdims=True)

            for i in range(self.dyna):
                s = rand.randint(0, self.num_states-1)
                a = rand.randint(0, self.num_actions-1)
                new_s = np.argmax(norm_tc[s, a, :])
                r = self.r[s, a]
                self.update_q(s, a, new_s, r)

        self.t_c[self.s, self.a, s_prime] += 1
        self.r[self.s, self.a] *= 1 - self.alpha
        self.r[self.s, self.a] += r * self.alpha
        simulate()

    def author(self):
        return 'mwong83'  # replace tb34 with your Georgia Tech username.


if __name__ == "__main__":
    print("Remember Q from Star Trek? Well, this isn't him")
