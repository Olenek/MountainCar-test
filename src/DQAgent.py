from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
# from tqdm.notebook import tqdm
from tqdm import tqdm

import numpy as np
import gym
import random


class DQAgent:
    def __init__(self, alpha, gamma, epsilon, epsilon_decay_speed=5, saving_path="./models"):
        """ action space: 0 is push left, 1 is  no push and 2 is push right
            state space: [x, v]; x \in [-1.2; 0.6]; v \in [-0.07, 0.07]
        """
        self._env = gym.make('MountainCar-v0')
        self._memory = []
        self._memory_load = 0
        self.gamma = gamma
        self.learning_rate = alpha
        self.model = self._build_model()
        self.epsilon = epsilon
        self.eps_d_s = epsilon_decay_speed
        self.batch_size = 128
        self.saving_path = saving_path

    def _build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=2, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(3, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def save_model(self, model_name):
        path = self.saving_path + '/' + model_name
        self.model.save(path)
        return 0

    def load_model(self, model_name):
        path = self.saving_path + '/' + model_name
        self.model = load_model(path)
        return 0

    def _memorize(self, state, action, reward, next_state, done):
        self._memory.append([state, action, reward, next_state, done])
        self._memory_load = len(self._memory)

    def _clear_memory(self):
        self._memory = []
        self._memory_load = len(self._memory)

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return self._env.action_space.sample()

        return np.argmax(self.model.predict(state)[0])

    def _replay(self):
        if self.batch_size > self._memory_load:
            return -1
        batch = np.array(random.sample(self._memory, self.batch_size), dtype=object)
        states, actions, rewards, next_states, dones = np.hsplit(batch, 5)
        states = np.concatenate((np.squeeze(states[:])), axis=0)
        actions = actions.reshape(self.batch_size, ).astype(int)
        rewards = rewards.reshape(self.batch_size, ).astype(float)
        next_states = np.concatenate(np.concatenate(next_states))
        dones = np.concatenate(dones).astype(bool)
        undones = ~ dones
        undones = undones.astype(float)
        targets = self.model.predict(states)
        q_futures = self.model.predict(next_states).max(axis=1)
        targets[(np.arange(self.batch_size), actions)] = rewards * dones + (rewards + q_futures * self.gamma) * undones
        self.model.fit(states, targets, epochs=1, verbose=0)
        return 0

    def _run_one_episode(self, initial_state):
        total_reward = 0
        max_height = -1e10
        current_state = initial_state
        done = False
        while not done:
            action = self.choose_action(current_state)
            next_state, reward, done, _ = self._env.step(action)
            max_height = max(max_height, next_state[0])
            next_state = next_state.reshape(1, 2)
            modified_reward = reward + 100 * (self.gamma * abs(next_state[0][1]) - abs(current_state[0][1]))
            self._memorize(current_state, action, modified_reward, next_state, done)
            total_reward += reward
            current_state = next_state
            self._replay()

        return total_reward, max_height

    def train(self, episodes, logging=False):
        global_max_score = -1e10
        global_max_height = -1e10

        for episode in tqdm(range(episodes)):
            initial_state = self._env.reset().reshape(1, 2)
            total_score, max_height = self._run_one_episode(initial_state)
            if logging:
                print("Episode: {}".format(episode))
                print(" Total score for episode {} : {}, max height : {}".format(episode, total_score, max_height))
                print(" GLOBAL MAXIMA: max score : {}, max height  : {}".format(global_max_score, global_max_height))
                print('-' * 150)
            global_max_score = max(global_max_score, total_score)
            global_max_height = max(global_max_height, max_height)
            self.epsilon -= 2 / episodes if self.epsilon > 0 else 0  # epsilon reduction
        return global_max_score, global_max_height

    def test_one(self, render=False):
        state = self._env.reset()
        done = False
        step = 0
        while not done:
            self._env.render() if render else 0
            state = state.reshape(1, 2)
            action = self.choose_action(state)
            step += 1
            next_state, reward, done, _ = self._env.step(action)
            state = next_state

            # end if solved
            if done and step < 200:
                print("Climbed in {} steps".format(step))
                return 0
        print("Task failed")
        return 0
