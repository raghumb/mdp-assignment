import numpy as np
import gym
import random

class QLearning:
    def __init__(self, environment, gamma, alpha, epsilon, num_episodes, stopping_epsilon):
        self.environment = environment
        self.discount_rate = gamma
        self.learning_rate = alpha
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.stopping_epsilon = stopping_epsilon


    def test_run(self):
        env = self.environment
        # Init arbitary values
        action_space_size = env.action_space.n
        state_space_size = env.observation_space.n
        q_table = np.zeros((state_space_size, action_space_size))
        max_steps_per_episode = 100

        # Hyperparameters

        exploration_rate = 1
        max_exploration_rate = 1
        min_exploration_rate = 0.01
        exploration_decay_rate = self.epsilon

        episode_lengths = np.zeros(self.num_episodes)
        episode_rewards = np.zeros(self.num_episodes)
        rewards_all_episodes = []
        prev_q_val = 0.
        q_values_first = []
        q_values_last = []
        self.rewards_first = []
        self.rewards_last = []
        # Q-learning algorithm
        
        for episode in range(self.num_episodes):
            # initialize new episode params
            state = env.reset()
            done = False
            rewards_current_episode = 0
            step = -1
            #for step in range(max_steps_per_episode): 
            while not done:
                step = step + 1
                # Exploration-exploitation trade-off
                # Take new action
                # Update Q-table
                # Set new state
                # Add new reward  

                # Exploration-exploitation trade-off
                exploration_rate_threshold = random.uniform(0, 1)
                if exploration_rate_threshold > exploration_rate:
                    action = np.argmax(q_table[state,:]) 
                else:
                    action = env.action_space.sample()     

                new_state, reward, done, info = env.step(action)
                episode_lengths[episode] = step
                episode_rewards[episode] += reward

                # Update Q-table for Q(s,a)
                new_q_value = q_table[state, action] * (1 - self.learning_rate) + \
                    self.learning_rate * (reward + self.discount_rate * np.max(q_table[new_state, :])) 

                q_table[state, action] = new_q_value
                if episode == 0:
                    q_values_first.append(new_q_value)
                    self.rewards_first.append(reward)
                if episode == self.num_episodes -1:
                    print('last episode')
                    q_values_last.append(new_q_value) 
                    self.rewards_last.append(reward)                   

                state = new_state
                rewards_current_episode += reward 

                if done == True: 
                    print('converged')
                    break

                #print('prev_q_val: ' + str(prev_q_val))
                #print('new_q_value: ' + str(new_q_value))


                prev_q_val = new_q_value

            exploration_rate = min_exploration_rate + \
                (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)

            # Exploration rate decay   
            # Add current episode reward to total rewards list
            rewards_all_episodes.append(rewards_current_episode)

            """if episode > 100 and prev_q_val!=0. and new_q_value !=0. and np.abs(prev_q_val - new_q_value) < self.stopping_epsilon:
                print('episode value for con: '+ str(episode))
                print('Converged!!')
                break """           

        return rewards_all_episodes, episode_lengths, episode_rewards,q_values_first, q_values_last


if __name__ == '__main__':
    #env_name  = 'FrozenLake-v0'    
    env_name  = 'FrozenLake8x8-v0'    
    gamma = 0.99
    alpha = 0.1
    epsilon = 0.000001
    stopping_epsilon =  1e-25 #0.0008
    env = gym.make(env_name) 
    num_episodes = 1000000
    QL = QLearning(env, gamma, alpha, epsilon, num_episodes, stopping_epsilon)
    rewards_all_episodes, episode_lengths, episode_rewards = QL.test_run()
    # Calculate and print the average reward per thousand episodes
    #rewards_per_thosand_episodes = np.split(np.array(rewards_all_episodes),num_episodes/1000)
    count = 1000
    print('rewards_all_episodes')
    print(rewards_all_episodes)

    print("********Average reward per thousand episodes********\n")
    """for r in rewards_per_thosand_episodes:
        print(count, ": ", str(sum(r/1000)))
        count += 1000"""
