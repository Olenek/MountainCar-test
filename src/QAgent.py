from tqdm import tqdm
# from tqdm.notebook import tqdm

import numpy as np
import gym
import random


class QAgent:
    def __init__(self, alpha, gamma, epsilon, epsilon_decay_speed=5, saving_path="./q-tables"):
        self._env = gym.make('MountainCar-v0')
        self.action_space = self._env.action_space.n  # 0 is push left, 1 is  no push and 2 is push right
        self.observation_space = self._env.observation_space  # [x, v]; x \in [-1.2; 0.6]; v \in [-0.07, 0.07]

        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # probability of choosing a random action

        self.v_states = np.linspace(-0.07, 0.07, num=20)
        self.x_states = np.linspace(-1.2, 0.6, num=20)
        self.states_size = len(self.v_states) * len(self.x_states)
        self.Q = np.zeros([self.states_size, self.action_space])
        self.saving_path = saving_path
        self.eps_d_s = epsilon_decay_speed

    def _get_Q_index(self, state):
        i = np.searchsorted(self.x_states, state[0], side="left")
        j = np.searchsorted(self.v_states, state[1], side="left")
        return len(self.v_states) * i + j

    def save_Q_table(self, model_name):
        path = self.saving_path + '/' + model_name
        return np.save(path + ".npy", self.Q)

    def load_Q_table(self, model_name):
        path = self.saving_path + '/' + model_name
        self.Q = np.load(path + ".npy")
        return 0

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return self._env.action_space.sample()

        return np.argmax(self.Q[state])

    def train(self, episodes, model_name, load_old=False, logging=False):
        if load_old:
            self.load_Q_table(model_name)

        global_max_score = -1e10
        global_max_height = -1e10
        episodes_to_solve = 0
        scores = []
        for i in tqdm(range(episodes)):
            obs = self._env.reset()
            state = self._get_Q_index(obs)
            done = False
            total_score = 0
            max_height = -1e10
            step = 0
            while not done:
                step += 1
                action = self.choose_action(state)

                next_obs, reward, done, info = self._env.step(action)
                modified_reward = reward + self.gamma * abs(next_obs[1]) - abs(obs[1])  # reward based on potentials
                next_state = self._get_Q_index(next_obs)

                # update Q
                self.Q[state, action] = (1 - self.alpha) * self.Q[state, action] + self.alpha * (
                        modified_reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state, action])
                state = next_state

                total_score += reward
                max_height = max(max_height, next_obs[0])

                # end if solved
                if done and step < 200:
                    if not episodes_to_solve:
                        episodes_to_solve = i
            scores.append(total_score)

            self.epsilon -= self.eps_d_s * self.epsilon / episodes if self.epsilon > 0 else 0  # epsilon reduction

            global_max_score = max(global_max_score, total_score)
            global_max_height = max(global_max_height, max_height)
            if i % 10 == 0 or (i == episodes - 1):
                if logging:
                    print("Episode: {}".format(i))
                    print(" Total score for episode {} : {}, max height : {}".format(i, total_score, max_height))
                    print(" GLOBAL MAXIMA: max score : {}, max height  : {}".format(global_max_score, global_max_height))
                    print('-' * 150)
                self.save_Q_table(model_name)

    def test_one(self, render=False):
        obs = self._env.reset()
        state = self._get_Q_index(obs)
        done = False
        step = 0
        while not done:
            self._env.render() if render else 0
            action = np.argmax(self.Q[state])
            step += 1
            next_obs, reward, done, info = self._env.step(action)
            next_state = self._get_Q_index(next_obs)

            state = next_state

            # end if solved
            if done and step < 200:
                print("Climbed in {} steps".format(step))
                return 0
        print("Task failed")
        return 0
