from algorithms.PolicyIter import PolicyIter
from algorithms.ValueIter import ValueIter
from problems.FrozenLake import FrozenLake
from problems.Taxi import Taxi
from algorithms.QLearning import QLearning
from algorithms.QL2 import QL2
from plotter import plot_curve, plot_rewards, plot_value_fn,plot_curve_single
import numpy as np
import pandas as pd
import time

class Experiment:
    def __init__(self):
        self.num_states = 8
        pass

    def experiment(self):
        self.experiment_1()
        self.experiment_2()        
        
        self.experiment_3()
        self.experiment_4()
        self.experiment_5()  

    def experiment_1(self):
        env_name = ''
        max_iter = 1000000
        gamma = 1.0
        avg_scores = []
        converge_iters = []
        policy_scores = []
        iter_arr = []
        gamma_range = np.arange(0.1, 1.0, 0.1)
        print('here')
        eps = 1e-25
        df = pd.DataFrame(columns=['gamma','state','value'])
        times = []
        for gamma in gamma_range:
            #fl = FrozenLake(self.num_states)
            fl = Taxi()
            print('gamma '+ str(gamma))
            
            PI = PolicyIter(fl.get_env(), gamma, max_iter, eps)
            start = time.time()
            optimal_policy, converge_iter, v_arr, avg_score = PI.test_run()
            end = time.time()
            times.append(end - start)
            print('avg reward' + str(avg_score))
            print('converge_iters' + str(converge_iter))
            avg_scores.append(avg_score)
            converge_iters.append(converge_iter)
            """df = df.append(pd.DataFrame({'gamma':[gamma for i in range(0,fl.get_env().observation_space.n)],
                                'state':[i for i in range(0,fl.get_env().observation_space.n)],
                                'value': V}))"""
            
        
        title = 'Reward Vs Gamma'
        x_title = 'Gamma'
        y_title = 'Average Reward'
        plot_curve(gamma_range, avg_scores, title, x_title, y_title, prefix = 'PolicyIter') 
        
        title = 'Convergence Vs Gamma'
        x_title = 'Gamma'
        y_title = 'Convergence Step'
        plot_curve(gamma_range, converge_iters, title, x_title, y_title, prefix = 'PolicyIter')        
        
        title= 'Values per Iteration'
        #fl = FrozenLake(self.num_states)
        fl = Taxi()
        gamma = 0.5            
        PI = PolicyIter(fl.get_env(), gamma, max_iter, eps)
        optimal_policy, converge_iter, v_arr, avg_score = PI.test_run()        
        plot_curve_single(v_arr, title, 'Iteration', 'Value', prefix = 'PolicyIter')        
        
        title = 'Time Complexity Vs Gamma'
        x_title = 'Gamma'
        y_title = 'Time (seconds)'
        plot_curve(gamma_range, times, title, x_title, y_title, prefix = 'PolicyIter')        


    def experiment_2(self):
        env_name = ''
        max_iter = 100000
        policy_scores = []
        converge_iters = []
        gamma_range = np.arange(0.1, 1.0, 0.1)
        print('here in experiment_2')
        eps = 1e-20
        df = pd.DataFrame(columns=['gamma','state','value'])
        times = []
        for gamma in gamma_range:
            #fl = FrozenLake(self.num_states)
            fl = Taxi()
            
            PI = ValueIter(fl.get_env(), gamma, max_iter, eps)
            start = time.time()
            policy_score, converge_iter, v_arr, V = PI.test_run()
            end = time.time()
            times.append(end - start)
            converge_iters.append(converge_iter)
            policy_scores.append(policy_score)
            
            df = df.append(pd.DataFrame({'gamma':[gamma for i in range(0,fl.get_env().observation_space.n)],
                                'state':[i for i in range(0,fl.get_env().observation_space.n)],
                                'value': V}))
            
        
        df.state=df.state.astype(int)
        
        #plot_value_fn(df, 'Value Iteration - Values per gamma')
        
        
        title = 'Reward Vs Gamma'
        x_title = 'Gamma'
        y_title = 'Average Reward'
        plot_curve(gamma_range, policy_scores, title, x_title, y_title, prefix = 'ValueIter') 
        
        title = 'Convergence Vs Gamma'
        x_title = 'Gamma'
        y_title = 'Convergence Step'
        plot_curve(gamma_range, converge_iters, title, x_title, y_title, prefix = 'ValueIter')        

     
        
        title= 'Values per Iteration'
        #fl = FrozenLake(self.num_states)
        fl = Taxi()
        gamma = 0.5            
        PI = ValueIter(fl.get_env(), gamma, max_iter, eps)
        policy_score, converge_iter, v_arr, V = PI.test_run()        
        plot_curve_single(v_arr, title, 'Iteration', 'Value', prefix = 'ValueIter')        
        
        title = 'Time Complexity Vs Gamma'
        x_title = 'Gamma'
        y_title = 'Time (seconds)'
        plot_curve(gamma_range, times, title, x_title, y_title, prefix = 'ValueIter')        


    def experiment_5(self):
        gamma = 0.5
        eps = 1e-25
        max_iter = 1000000
        fl = Taxi()
            
        PI = PolicyIter(fl.get_env(), gamma, max_iter, eps)
        start = time.time()
        optimal_policy_PI, converge_iter, v_arr, avg_score = PI.test_run()
        end = time.time()
        print('optimal policy- PI')
        #print(optimal_policy_PI)
        print('Time diff: '+ str(end - start))
        print('converge_iter '+ str(converge_iter) )

        
        eps = 1e-20
        fl = Taxi()            
        PI = ValueIter(fl.get_env(), gamma, max_iter, eps)
        start = time.time()
        optimal_policy_VI, converge_iter = PI.test_run_w_policy()
        end = time.time()
        print('optimal policy- VI')
        #print(optimal_policy_VI)  
        print('Time diff: '+ str(end - start))
        print('converge_iter '+ str(converge_iter) )
        if (np.all(optimal_policy_PI == optimal_policy_VI)):   
            print('policies match')

        
       


    def experiment_4(self):
        env_name = ''
        #fl = FrozenLake(self.num_states)
        fl = Taxi()
        gamma = 0.6
        alpha = 0.1
        epsilon = 0.1 #0.0000001
        num_episodes = 100000
        stopping_epsilon =  0.0008
        
        
        #fl = FrozenLake(self.num_states)
        fl = Taxi() 
        QL = QLearning(fl.get_env(), gamma, alpha, epsilon, num_episodes, stopping_epsilon)
        rewards_all_episodes, episode_lengths, episode_rewards,q_values_first, q_values_last = QL.test_run()
        rewards_first = QL.rewards_first
        rewards_last = QL.rewards_last
        rewards_first = np.cumsum(rewards_first)
        rewards_last = np.cumsum(rewards_last)
        avg_length = np.mean(episode_lengths)
        avg_rewards = np.mean(episode_rewards)
        print('Average Convergence Length '+ str(avg_length))
        print('Average Rewards '+ str(avg_rewards))
        #print('rewards_all_episodes')
        #print(rewards_all_episodes)
        # Calculate and print the average reward per thousand episodes
        """rewards_per_thosand_episodes = np.split(np.array(rewards_all_episodes),num_episodes/1000)
        count = 1000

        print(episode_rewards)
        rewards_arr = []
        counts_arr = []
        print("********Average reward per thousand episodes********\n")
        for r in rewards_per_thosand_episodes:
            print(count, ": ", str(sum(r/1000)))
            rewards_arr.append(sum(r/1000))
            count += 1000   
            counts_arr.append(count)  """
       

        plot_curve_single(q_values_first, 'QValues  Vs Step (1st Episode)', "Step", "QValues", prefix = 'QLTaxi')
        plot_curve_single(q_values_last, 'QValues  Vs Step(Last Episode)', "Step", "QValues", prefix = 'QLTaxi')
        #plot_curve(counts_arr, rewards_arr,  'Episode Reward over Time', "Episode", "Episode Reward", prefix = None)         
        plot_curve_single(episode_rewards, 'Episode Reward Vs Episode', "Episode", "Episode Reward", prefix = 'QLTaxi')

        plot_curve_single(episode_lengths, 'Convergence Per Episode', "Episode", "Steps For Convergence", prefix = 'QLTaxi')
        plot_curve_single(rewards_first, 'Rewards Vs Steps(1st Episode)', "Step", "Rewards", prefix = 'QLTaxi')
        plot_curve_single(rewards_last, 'Rewards Vs Steps(Last Episode)', "Step", "Rewards", prefix = 'QLTaxi')



    def experiment_3(self):
        env_name = ''
        #fl = FrozenLake(self.num_states)
        fl = Taxi()        
        alpha = 0.1
        epsilon = 0.1 #0.0000001
        num_episodes = 100000
        stopping_epsilon =  0.0008
        gamma_range = np.arange(0.1, 1.1, 0.1)
        avg_lengths = []
        avg_rewards = []
        times = []
        for gamma in gamma_range:
            #fl = FrozenLake(self.num_states)
            fl = Taxi() 
            QL = QLearning(fl.get_env(), gamma, alpha, epsilon, num_episodes, stopping_epsilon)
            start = time.time()
            rewards_all_episodes, episode_lengths, episode_rewards,q_values_first, q_values_last = QL.test_run()
            end = time.time()
            times.append(end - start)
            avg_length = np.mean(episode_lengths)
            avg_reward = np.mean(episode_rewards)
            avg_lengths.append(avg_length)
            avg_rewards.append(avg_reward)
            #print('rewards_all_episodes')
            #print(rewards_all_episodes)
            # Calculate and print the average reward per thousand episodes
              

        plot_curve(gamma_range, avg_lengths, 'Convergence Steps Vs Gamma', 'Gamma', 'Convergence Steps', prefix = 'QL-Taxi')        
        plot_curve(gamma_range, avg_rewards, 'Cumulative Rewards Vs Gamma', 'Gamma', 'Cumulative Rewards', prefix = 'QL-Taxi')        
        title = 'Time Complexity Vs Gamma'
        x_title = 'Gamma'
        y_title = 'Time (seconds)'
        plot_curve(gamma_range, times, title, x_title, y_title, prefix = 'QL-Taxi')         

if __name__ == '__main__':
    exp = Experiment()
    exp.experiment()