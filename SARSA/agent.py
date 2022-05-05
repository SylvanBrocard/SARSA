"""
Agent class for the SARSA algorithm.
"""

from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Tuple

import numpy as np
import tensorflow as tf

from .maze import Maze, MazeWithGhosts

start_Q = 0
start_Q_for_exits = 0


class Agent(ABC):
    """Abstract agent class."""

    def __init__(self, maze: Maze, seed=None):
        """
        Initialize the agent.

        Parameters
        ----------
        maze : maze.Maze
            The environment to interact with.
        seed : int
            The seed for the random number generator.
        """
        self.maze = maze
        self.rng = np.random.default_rng(seed=seed)

    @abstractmethod
    def act(self) -> int:
        """
        Choose an action.

        Returns
        -------
        action : int
            The action to take.
        """
        pass

    @abstractmethod
    def learn(
        self,
        reward: int,
        state: Tuple,
        action: int,
        state_prime: Tuple,
        action_prime: int,
    ):
        """
        Update the agent's knowledge of the environment.

        Parameters
        ----------
        reward : float
            The reward received.
        state : namedtuple
            The current state.
        action : int
            The action taken.
        state_prime : namedtuple
            The next state.
        action_prime : int
            The next action.
        """
        pass

    def train(self, replay_buffer: list):
        """
        Train the agent.

        Parameters
        ----------
        replay_buffer : list
            The replay buffer.
        """
        pass


class RandomAgent(Agent):
    """Random agent."""

    def act(self):
        eligible_actions = self.maze.eligible_actions(*self.maze.current_position)
        return self.rng.choice(eligible_actions)

    def learn(
        self,
        reward: int,
        state: Tuple,
        action: int,
        state_prime: Tuple,
        action_prime: int,
    ):
        pass


class SARSAAgent(Agent):
    """SARSA agent."""

    def __init__(self, maze: Maze, alpha=0.1, gamma=0.9, epsilon=0.1, seed=None):
        """
        Initialize the agent.

        Parameters
        ----------
        maze : maze.Maze
            The environment to interact with.
        alpha : float
            The learning rate.
        gamma : float
            The discount factor.
        epsilon : float
            The probability of choosing a random action.
        seed : int
            The seed for the random number generator.
        """
        super().__init__(maze, seed)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = None
        self.initialize_Q()

    def initialize_Q(self):
        """
        Initialize the Q matrix.
        """
        self.Q = (
            np.ones((self.maze.shape[0], self.maze.shape[1], 4), dtype=np.float)
            * start_Q
        )

        # set exits to some value
        exits = self.maze.get_exits()
        self.Q[exits] = start_Q_for_exits

    def act(self) -> int:
        """
        Choose an action with e-greedy.

        Returns
        -------
        action : int
            The action to take.
        """

        # check eligible moves.
        eligible_actions = self.maze.eligible_actions(*self.maze.current_position)
        if len(eligible_actions) == 0:
            return -1

        # decide whether to take a random action or not
        if self.rng.uniform() < self.epsilon:
            action = self.rng.choice(eligible_actions)
        else:
            action_values = self.Q[
                self.maze.current_position[0], self.maze.current_position[1], :
            ]
            max_value_actions = np.nonzero(
                action_values == np.max(action_values[eligible_actions])
            )[0]
            max_value_actions = max_value_actions[
                np.in1d(max_value_actions, eligible_actions)
            ]
            action = self.rng.choice(max_value_actions)

        return action

    def learn(
        self,
        reward: int,
        state: Tuple,
        action: int,
        state_prime: Tuple,
        action_prime: int,
    ):
        """
        Update the agent's knowledge of the environment.

        Parameters
        ----------
        reward : float
            The reward received.
        state : namedtuple
            The current state.
        action : int
            The action taken.
        state_prime : namedtuple
            The next state.
        action_prime : int
            The next action.
        """

        # get current and previous positions
        x_new, y_new = state_prime[0]
        x_old, y_old = state[0]

        # update knowledge
        QSA_prime = self.Q[x_new, y_new, action_prime]
        QSA = self.Q[x_old, y_old, action]
        self.Q[x_old, y_old, action] += self.alpha * (
            reward + self.gamma * QSA_prime - QSA
        )


class DeepAgent(Agent):
    """Agent with Deep Q learning"""

    def __init__(self, maze: Maze, alpha=0.1, gamma=0.9, epsilon=0.1, min_len_buffer=50, seed=None):
        """
        Initialize the agent.

        Parameters
        ----------
        maze : maze.Maze
            The environment to interact with.
        alpha : float
            The learning rate.
        gamma : float
            The discount factor.
        epsilon : float
            The probability of choosing a random action.
        seed : int
            The seed for the random number generator.
        """
        super().__init__(maze, seed)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_len_buffer = min_len_buffer
        self.has_trained_model = False

        self.model = tf.keras.models.Sequential(
            [
                # tf.keras.Input(shape=(2*(self.maze.nb_ghosts+1),), name="input"),
                tf.keras.layers.Dense(30, activation="relu", input_shape=(2 * (self.maze.nb_ghosts + 1),)),
                tf.keras.layers.Dense(30, activation="relu"),
                tf.keras.layers.Dense(4, activation="linear"),
            ]
        )
        self.model.compile(optimizer="adam", loss="rmse")

    def act(self, state: Tuple) -> int:
        """
        Choose an action with e-greedy.

        Parameters
        ----------
        state : Tuple
            The current state.

        Returns
        -------
        action : int
            The action to take.
        """
        if self.has_trained_model:
            action = np.argmax(self.model.predict(np.array(state)))
        else:
            eligible_actions = self.maze.eligible_actions(*self.maze.current_position)
            if len(eligible_actions) == 0:
                return -1
            action = self.rng.choice(eligible_actions)
        return action

    def learn(
        self,
        reward: int,
        state: Tuple,
        action: int,
        state_prime: Tuple,
        action_prime: int,
    ):
        """
        Update the agent's knowledge of the environment.

        Parameters
        ----------
        reward : float
            The reward received.
        state : namedtuple
            The current state.
        action : int
            The action taken.
        state_prime : namedtuple
            The next state.
        action_prime : int
            The next action.
        state_history : list
            The history of states.
        """
        pass

    def train(self, replay_buffer: list):
        """
        Train the agent.

        Parameters
        ----------
        replay_buffer : list
            The replay buffer.
        """
        if len(replay_buffer) < self.min_len_buffer:
            return
        (
            reward_batch,
            state_batch,
            action_batch,
            state_prime_batch,
            action_prime_batch,
            done_batch,
        ) = self.get_batch(replay_buffer)
        
        lst_q = []
        for state, action, reward, state_prime, done in zip(state_batch, action_batch, reward_batch, state_prime_batch, done_batch):
            if len(state) != 22:
                raise ValueError(f"state is empty: {state}")
            print(np.array(state))
            lst_q.append(self.model.predict(np.array(state)))

            if done:
                lst_q[-1][action] = reward
            else:
                expected_reward = self.model.predict(np.array(state_prime))
                idx_max = np.argmax(expected_reward)
                lst_q[-1][action] = reward + self.gamma * expected_reward[idx_max]

        self.model.fit(state_batch, lst_q, epochs=1, verbose=0)
        self.has_trained_model = True

    def get_batch(self, replay_buffer: list, batch_size: int = 50) -> list:
        """
        Get a batch of states.

        Parameters
        ----------
        state_history : list
            The history of states.
        batch_size : int
            The size of the batch.
        """
        history_size = len(replay_buffer)
        batch_indices = self.rng.choice(history_size, batch_size)

        reward_batch = [[replay_buffer[i][0] for i in batch_indices]]
        state_batch = [replay_buffer[i][1] for i in batch_indices]
        action_batch = [replay_buffer[i][2] for i in batch_indices]
        state_prime_batch = [replay_buffer[i][3] for i in batch_indices]
        action_prime_batch = [replay_buffer[i][4] for i in batch_indices]
        done_batch = [replay_buffer[i][5] for i in batch_indices]

        return (
            reward_batch,
            state_batch,
            action_batch,
            state_prime_batch,
            action_prime_batch,
            done_batch,
        )
