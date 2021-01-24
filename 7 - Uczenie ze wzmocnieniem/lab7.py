import numpy as np
import gym


class QLearning:
    def __init__(self, name, limit):
        self.limit = limit
        self.env = gym.make(name)
        self.env._max_episode_steps = limit
        self.q_val = np.zeros(self.env.env.nS * self.env.env.nA).reshape(self.env.env.nS, self.env.env.nA).astype(np.float32)

    def learn(self, alfa, gamma, itr):
        for i in range(itr):
            state = self.env.reset()
            done = False
            while not done:
                act = self.env.action_space.sample()
                next_state, reward, done, info = self.env.step(act)
                q_next_max = np.max(self.q_val[next_state])
                self.q_val[state][act] = (1 - alfa) * self.q_val[state][act] + alfa * (reward + gamma * q_next_max)
                state = next_state

    def test(self, itr):
        reward_total = 0.0
        for i in range(itr):
            state = self.env.reset()
            done = False
            while not done:
                act = np.argmax(self.q_val[state])
                next_state, reward, done, info = self.env.step(act)
                state = next_state
            reward_total += reward
        return reward_total / itr


if __name__ == "__main__":
    for alfa in [0.01, 0.1, 0.5, 1]:
        for gamma in [0.5, 1, 1.5]:
            agent = QLearning("FrozenLake8x8-v0", 200)
            print(f"###### LEARN {alfa}, {gamma} ######")
            agent.learn(alfa, gamma, 100000)
            avr_reward = agent.test(10000)
            print(f"average reward: {avr_reward:.2f}")
