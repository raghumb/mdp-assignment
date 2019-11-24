import numpy as np
import gym


#This part of code is inspired by https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa


class PolicyIter:
    def __init__(self, environment, gamma, max_iter, eps):
        self.environment = environment
        self.gamma = gamma
        self.max_iter = max_iter
        self.eps = eps

    def test_run(self): 
        print(' in PolicyIter')       
        
        optimal_policy, converge_iter, v_arr = self.policy_iteration(self.environment, gamma = self.gamma)
        scores = self.evaluate_policy(self.environment, optimal_policy, gamma = self.gamma)
        avg_score = np.mean(scores)
        print('Average scores = ', avg_score)
        return optimal_policy, converge_iter, v_arr, avg_score
    
    def run_episode(self, env, policy, gamma = 1.0, render = False):
        """ Runs an episode and return the total reward """
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


    def evaluate_policy(self, env, policy, gamma, n = 100):
        scores = [self.run_episode(env, policy, gamma, False) for _ in range(n)]
        return np.mean(scores)

    def extract_policy(self, v, env, gamma):
        """ Extract the policy given a value-function """
        policy = np.zeros(env.nS)
        for s in range(env.nS):
            q_sa = np.zeros(env.nA)
            for a in range(env.nA):
                q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in  env.P[s][a]])
            policy[s] = np.argmax(q_sa)
        return policy

    def compute_policy_v(self, env, policy, gamma):
        """ Iteratively evaluate the value-function under policy.
        Alternatively, we could formulate a set of linear equations in iterms of v[s] 
        and solve them to find the value function.
        """
        v = np.zeros(env.nS)
        
        while True:
            prev_v = np.copy(v)
            for s in range(env.nS):
                policy_a = policy[s]
                v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.P[s][policy_a]])
            
            v_diff = max(np.fabs(prev_v - v))
            if (v_diff <= self.eps):
                # value converged
                break
        return v

    def policy_iteration(self, env, gamma):
        """ Policy-Iteration algorithm """
        policy = np.random.choice(env.nA, size=(env.nS))  # initialize a random policy
        max_iterations = self.max_iter
        converge_iter = 0
        v_arr = []
        for i in range(max_iterations):
            old_policy_v = self.compute_policy_v(env, policy, gamma)
            
            value_sum = np.sum(old_policy_v)
            v_arr.append(value_sum)
            #print('value sum per iteration '+ str(value_sum))
            
            new_policy = self.extract_policy(old_policy_v, env, gamma)
            #print('new_policy ')
            #print(new_policy)
            if (np.all(policy == new_policy)):
                print ('Policy-Iteration converged at step %d.' %(i+1))
                converge_iter = i + 1
                break
            policy = new_policy
        return policy, converge_iter, v_arr


if __name__ == '__main__':
    env_name  = 'FrozenLake8x8-v0'
    gamma = 1.0
    eps = 1e-20
    max_iter = 100000
    env = gym.make(env_name)  
    PI = PolicyIter(env, gamma, max_iter, eps)
    optimal_policy, converge_iter, v_arr, avg_score = PI.test_run()
    print('avg reward' + str(avg_score))
    print('converge_iters' + str(converge_iter))