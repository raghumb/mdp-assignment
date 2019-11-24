import mdptoolbox.example
from mdptoolbox.mdp import ValueIteration, QLearning, PolicyIteration
import numpy as np
from algorithms.QLearningForest import QLearningForest

class ForestNG:
    def __init__(self, states, gamma, max_iter, epsilon, algo):
            self.states = states
            self.gamma = gamma
            self.max_iter = max_iter
            self.epsilon = epsilon
            self.algo = algo


    def test_run(self):
        P, R = mdptoolbox.example.forest(self.states)
        v = None
        iter = 0
        policy = None
        alg_impl = None
        if self.algo == 'ValueIteration':
            alg_impl = mdptoolbox.mdp.ValueIteration(P, R, self.gamma, self.epsilon, self.max_iter)
            alg_impl.run()
            policy = alg_impl.policy
            iter = alg_impl.iter
            v = alg_impl.V
            #print('policy: ')
            #print(policy)            
            #print('iter converged')
            #print(iter)
            #print('value function')
            #print(v)

        elif self.algo == 'PolicyIteration':
            alg_impl = mdptoolbox.mdp.PolicyIteration(P, R, self.gamma, None, self.max_iter)
            alg_impl.run()
            policy = alg_impl.policy
            iter = alg_impl.iter
            v = alg_impl.V
            #print('policy: ')
            #print(policy)            
            #print('iter converged')
            #print(iter)
            #print('value function')
            #print(v) 
                  
        elif self.algo == 'QLearning':
            alg_impl = QLearningForest(P, R, self.gamma, self.epsilon, self.max_iter)
            alg_impl.run()
            policy = alg_impl.policy
            
            v = alg_impl.V
            #print('policy: ')
            #print(policy)            

            ##print('value function')
            #print(v) 

        avg_value = np.sum(v)
        return policy, iter, avg_value, alg_impl


if __name__ == '__main__':
    eps = 1e-10
    forest = ForestNG(10, 0.2, 1000, eps, 'ValueIteration')
    forest.test_run()