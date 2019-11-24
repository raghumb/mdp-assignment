import numpy as np
import gym
import random

class QL3:
    def __init__(self, environment, gamma, alpha, epsilon, max_iter):
        self.environment = environment
        self.gamma = gamma
        self.max_iter = max_iter
        self.alpha = alpha
        self.epsilon = epsilon


    def policy_fn(self, state, n_action, Q): 
   
        a_prob = np.ones(n_action, dtype = float) * self.epsilon / n_action 
                  
        best_action = np.argmax(Q[state]) 
        a_prob[best_action] += (1.0 - self.epsilon) 
        return a_prob 
   

    #https://www.geeksforgeeks.org/q-learning-in-python/
    def test_run(self):
        env = self.environment
        # Number of possible actions
        action_size = env.action_space.n 
        print("Action size ", action_size) 


        # Number of possible states
        state_size = env.observation_space.n 
        print("State size ", state_size)
        qtable = np.zeros((state_size, action_size))
        # Init arbitary values
        episodes = 30000            # Total episodes
        max_steps = 1000            # Max steps per episode
        lr = 0.3                    # Learning rate
        decay_fac = 0.00001         # Decay learning rate each iteration
        gamma = 1               # Discounting rate - later rewards impact less
        rewards = np.zeros(episodes)
        lengths = np.zeros(episodes)
        for episode in range(episodes):
    
            state = env.reset() # Reset the environment
            done = False        # Are we done with the environment
            lr -= decay_fac     # Decaying learning rate
            step = 0
            
            if lr <= 0: # Nothing more to learn?
                break
                
            for step in range(max_steps):
                
                # Randomly Choose an Action
                a_prob = self.policy_fn(state, action_size, qtable)
                action = np.random.choice(np.arange( 
                                        len(a_prob)), 
                                        p = a_prob)                 
                
                # Take the action -> observe new state and reward
                new_state, reward, done, info = env.step(action)
                print('reward '+ str(reward))
                best_next_action = np.argmax(qtable[new_state]) 
                rewards[episode] += reward
                lengths[episode] += step

                td_target = reward + gamma * qtable[new_state][best_next_action] 
                td_delta = td_target - qtable[state][action] 
                qtable[state][action] += self.alpha * td_delta                 
 
                # if done.. jump to next episode
                if done == True:
                    break
                
                # moving states
                state = new_state
                
            episode += 1

        return rewards, lengths

if __name__ == '__main__':
    env_name  = 'FrozenLake8x8-v0'    
    gamma = 1.0
    env = gym.make(env_name) 
    epsilon = 0.1
    max_iter = 1000
    alpha = 0.6
    ql = QL3(env, gamma, alpha, epsilon, max_iter)
    r,l = ql.test_run()
    print(r)
    print(l)
            
