from algorithms.PolicyIter import PolicyIter
from algorithms.ValueIter import ValueIter
from problems.ForestNG import ForestNG
from algorithms.QLearning import QLearning
from algorithms.QL2 import QL2
from plotter import plot_curve, plot_curve_single
import numpy as np
import gym
import time

class ExperimentForest:
    def __init__(self):
        self.num_states = 5000
        self.max_iterations = 500000
        self.experiment()

    def experiment(self):
        self.experiment_1()
        self.experiment_2()        
        self.experiment_3()  
        self.experiment_4()  
        self.experiment_5()
        self.experiment_7()  

    def experiment_1(self):
        env_name = ''
        max_iter = self.max_iterations
        gamma = 1.0
        avg_scores = []
        converge_iters = []
        policy_scores = []
        iter_arr = []
        gamma_range = np.arange(0.1, 1.0, 0.1)
        print('here')
        eps = 1e-10        
        algo = 'PolicyIteration'
        times = []
        for gamma in gamma_range:
            print('gamma ' + str(gamma))
            fl = ForestNG(self.num_states, gamma, max_iter, eps, algo)
            start = time.time()
            policy, iter, avg_value, alg_impl = fl.test_run()
            end = time.time()
            times.append(end - start)
            avg_scores.append(avg_value)
            converge_iters.append(iter)
            print('avg score ' + str(avg_value))
            print('converge_iter ' + str(iter))
            
        title = 'Convergence Vs Gamma'
        x_title = 'Gamma'
        y_title = 'Convergence Step'
        plot_curve(gamma_range, converge_iters, title, x_title, y_title, prefix = 'PolicyIterForest')

        title = 'Total Value Vs Gamma'
        x_title = 'Gamma'
        y_title = 'Total Value'
        plot_curve(gamma_range, avg_scores, title, x_title, y_title, prefix = 'PolicyIterForest')        


        title = 'Time Complexity Vs Gamma'
        x_title = 'Gamma'
        y_title = 'Time (seconds)'
        plot_curve(gamma_range, times, title, x_title, y_title, prefix = 'PolicyIterForest')        


        """gamma = 0.4
        fl = ForestNG(states, gamma, max_iter, eps, algo)    
        policy, iter, avg_value = fl.test_run()
        policy_scores.append(policy_score)
        converge_iters.append(converge_iter)
        for i in range(len(v_arr)):
            iter_arr.append(i)

        title = 'Value Function Convergence'
        x_title = 'Iterations'
        y_title = 'Value Function'
        plot_curve(iter_arr, v_arr, title, x_title, y_title, prefix = 'PolicyIter')
        """

    def experiment_2(self):
        env_name = ''
        max_iter = self.max_iterations
        policy_scores = []
        converge_iters = []
        gamma_range = np.arange(0.1, 1.1, 0.1)
        print('here')
        eps = 1e-10        
        algo = 'ValueIteration'
        times = []
        for gamma in gamma_range:
            fl = ForestNG(self.num_states, gamma, max_iter, eps, algo)
            start = time.time()
            policy, converge_iter, avg_value, alg_impl = fl.test_run()
            end = time.time()
            times.append(end - start)
            policy_scores.append(avg_value)
            converge_iters.append(converge_iter) 

        title = 'Convergence Vs Gamma'
        x_title = 'Gamma'
        y_title = 'Convergence Step'
        plot_curve(gamma_range, converge_iters, title, x_title, y_title, prefix = 'ValueIterForest')

        title = 'Total Value Vs Gamma'
        x_title = 'Gamma'
        y_title = 'Total Value'
        plot_curve(gamma_range, policy_scores, title, x_title, y_title, prefix = 'ValueIterForest')        

        title = 'Time Complexity Vs Gamma'
        x_title = 'Gamma'
        y_title = 'Time (seconds)'
        plot_curve(gamma_range, times, title, x_title, y_title, prefix = 'ValueIterForest') 

        """iter_arr = []
        gamma = 0.4
        fl = FrozenLake(10)
        VI = ValueIter(fl.get_env(), gamma, max_iter, eps)
        policy_score, converge_iter, v_arr = VI.test_run()
        policy_scores.append(policy_score)
        converge_iters.append(converge_iter)
        for i in range(len(v_arr)):
            iter_arr.append(i)

        title = 'Value Function Convergence'
        x_title = 'Iterations'
        y_title = 'Value Function'
        plot_curve(iter_arr, v_arr, title, x_title, y_title, prefix = 'ValueIterForest')

        """



    def experiment_3(self):
        env_name = ''
        max_iter = self.max_iterations
        gamma = 0.6
        alpha = 0.1
        epsilon = 0.1
        times = []
        #QL = QLearning(fl.get_env(), gamma, alpha, epsilon, max_iter)
        algo = 'QLearning'        
        fl = ForestNG(self.num_states, gamma, max_iter, epsilon, algo)
        start = time.time()
        policy, iter, avg_value, alg_impl = fl.test_run()
        end = time.time()
        diff = end - start
        print('ql time '+ str(diff))
        rewards_cum = np.cumsum(alg_impl.rewards)
        md = alg_impl.mean_discrepancy
        #index = md.index(min(md))
        #print('index '+ str(index))
        avg_qvar = np.mean(md)
        print('avg_qvar '+ str(avg_qvar))

        plot_curve(alg_impl.mean_iter_arr, alg_impl.mean_discrepancy, 'QValue Variance Vs Iteration',  'Iteration','QValue Variance (Average)', prefix = 'QL-Forest')
        plot_curve_single(rewards_cum, 'Rewards Vs Steps', "Step", "Rewards", prefix = 'QL-Forest')    

    def experiment_4(self):
        env_name = ''
        max_iter = self.max_iterations #100000
        
        alpha = 0.1
        epsilon = 0.1
        #QL = QLearning(fl.get_env(), gamma, alpha, epsilon, max_iter)
        algo = 'QLearning'     
        gamma_range = np.arange(0.1, 1.1, 0.1) 
        avg_rewards = []
        avg_qvars = []
        times = []
        for gamma in gamma_range:  
            fl = ForestNG(self.num_states, gamma, max_iter, epsilon, algo)
            start = time.time()
            policy, iter, avg_value, alg_impl = fl.test_run()
            end = time.time()
            times.append(end - start)
            md = alg_impl.mean_discrepancy
            #index = md.index(min(md))
            #print('index '+ str(index))
            avg_reward = np.mean(alg_impl.rewards)
            avg_rewards.append(avg_reward)
            avg_qvar = np.mean(md)
            avg_qvars.append(avg_qvar)

        #plot_curve(alg_impl.mean_iter_arr, alg_impl.mean_discrepancy, 'Averge Delta Vs Iteration',  'Iteration','Average Delta', prefix = 'QL-Forest')
        title = 'Avg Reward Vs Gamma'
        x_title = 'Gamma'
        y_title = 'Average Reward'
        plot_curve(gamma_range, avg_rewards, title, x_title, y_title, prefix = 'QL-Forest')        

        title = 'Avg Q Variance Vs Gamma'
        x_title = 'Gamma'
        y_title = 'Avg Q Variance'
        plot_curve(gamma_range, avg_rewards, title, x_title, y_title, prefix = 'QL-Forest')        

        title = 'Time Complexity Vs Gamma'
        x_title = 'Gamma'
        y_title = 'Time (seconds)'
        plot_curve(gamma_range, times, title, x_title, y_title, prefix = 'QL-Forest')


    def experiment_5(self):
        max_iter = self.max_iterations
        eps = 1e-10
        gamma = 0.5
        algo = 'ValueIteration'
        fl = ForestNG(self.num_states, gamma, max_iter, eps, algo)
        start = time.time()
        policy_VI, converge_iter, avg_value, alg_impl = fl.test_run()
        end = time.time()
        print('diff ' + str(end - start))
        print('policy ')
        print('converge_iter')
        print(converge_iter)
        #print(policy_VI)

        algo = 'PolicyIteration'
        fl = ForestNG(self.num_states, gamma, max_iter, eps, algo)
        start = time.time()
        policy_PI, iter, avg_value, alg_impl = fl.test_run()
        end = time.time()
        print('diff ' + str(end - start))
        print('policy_PI ')
        print('converge_iter')
        print(converge_iter)
        #print(policy_PI)        
        if (np.all(policy_VI == policy_PI)):   
            print('policies match')



    def experiment_7(self):
        env_name = ''
        max_iter = self.max_iterations
        gamma = 0.6
        alpha = 0.1
        epsilon = 0.1
        eps = 0.1
        times = []
        algo = 'QLearning'
        fl = ForestNG(self.num_states, gamma, max_iter, eps, algo)
        start = time.time()
        policy_PI, iter, avg_value, alg_impl = fl.test_run()
        plot_curve_single(alg_impl.episode_rewards, 'Episode Reward Vs Episode', "Episode", "Episode Reward", prefix = 'QLForest')

if __name__ == '__main__':
    exp = ExperimentForest()
    exp.experiment()