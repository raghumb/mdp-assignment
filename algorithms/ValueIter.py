import numpy as np
import gym


#This part of code is inspired by https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa

class ValueIter:
    def __init__(self, environment, gamma, max_iter, eps):
        self.environment = environment
        self.gamma = gamma
        self.max_iter = max_iter
        self.eps = eps


    def test_run(self): 
        optimal_v, converge_iter, v_arr = self.value_iteration(self.environment, self.gamma, self.max_iter)
        policy = self.extract_policy(optimal_v, self.environment, self.gamma)
        policy_score = self.evaluate_policy(self.environment, policy, self.gamma, n=self.max_iter)
        print('Policy average score = ', policy_score)
        return policy_score, converge_iter, v_arr, optimal_v

    
    def test_run_w_policy(self):
        optimal_v, converge_iter, v_arr, r_arr = self.value_iteration(self.environment, self.gamma, self.max_iter)
        policy = self.extract_policy(optimal_v, self.environment, self.gamma)
        policy_score = self.evaluate_policy(self.environment, policy, self.gamma, n=self.max_iter)
        print('Policy average score = ', policy_score)
        return policy, converge_iter



#This part of code is inspired by https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa
    def value_iteration(self, env,gamma = 1.0, max_iterations = 100000):
        """ Value-iteration algorithm """
        v = np.zeros(env.nS)  # initialize value-function
        
        converge_iter = 0
        v_arr = []
        r_arr = []
        for i in range(max_iterations):
            prev_v = np.copy(v)
            for s in range(env.nS):
                ###q_sa = np.zeros(env.action_space.n)
                q_sa = [sum([p*(r + gamma * prev_v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env.nA)]                
                        
                v[s] = np.max(q_sa)
                
            
            value_sum = np.sum(v)
            v_arr.append(value_sum)
            #print('value sum per iteration '+ str(value_sum))
            v_diff = max(np.fabs(prev_v - v))
            #print('v_diff ' + str(v_diff))
            if (v_diff <= self.eps):
                print ('Value-iteration converged at iteration# %d.' %(i+1))
                converge_iter = i + 1
                break
        return v, converge_iter, v_arr

    def extract_policy(self, v, env, gamma = 1.0):
        """ Extract the policy given a value-function """
        policy = np.zeros(env.nS)
        for s in range(env.nS):
            q_sa = np.zeros(env.action_space.n)
            for a in range(env.action_space.n):
                for next_sr in env.P[s][a]:
                    # next_sr is a tuple of (probability, next state, reward, done)
                    p, s_, r, _ = next_sr
                    q_sa[a] += (p * (r + gamma * v[s_]))
            policy[s] = np.argmax(q_sa)
        return policy

    def evaluate_policy(self, env, policy, gamma = 1.0,  n = 100):
        """ Evaluates a policy by running it n times.
        returns:
        average total reward
        """
        scores = [
                self.run_episode(env, policy, gamma = gamma, render = False)
                for _ in range(n)]
        return np.mean(scores)


    def run_episode(self, env, policy, gamma = 1.0, render = False):
        """ Evaluates policy by using it to run an episode and finding its
        total reward.
        args:
        env: gym environment.
        policy: the policy to be used.
        gamma: discount factor.
        render: boolean to turn rendering on/off.
        returns:
        total reward: real value of the total reward recieved by agent under policy.
        """
        obs = env.reset()
        total_reward = 0
        step_idx = 0
        while True:
            if render:
                env.render()
            obs, reward, done , _ = env.step(int(policy[obs]))
            total_reward += (gamma ** step_idx * reward)
            step_idx += 1
            if done:
                break
        return total_reward

if __name__ == '__main__':
    env_name  = 'FrozenLake8x8-v0'
    gamma = 1.0
    eps = 1e-20
    max_iter = 100000
    env = gym.make(env_name)
    VI = ValueIter(env, gamma, max_iter, eps)
    VI.test_run()
    """optimal_v, converge_iter = VI.value_iteration(env, gamma, max_iter)
    policy = VI.extract_policy(optimal_v, env, gamma)
    policy_score = VI.evaluate_policy(env, policy, gamma, n=100)
    print('Policy average score = ', policy_score)"""
    
    