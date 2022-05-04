"""
Agent class for the SARSA algorithm.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from maze import Maze

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
    def learn(self, reward, state, action, state_prime, action_prime):
        """
        Update the agent's knowledge of the environment.

        Parameters
        ----------
        action : int
            The action taken.
        reward : float
            The reward received.
        """
        pass


class RandomAgent(Agent):
    """Random agent."""

    def act(self):
        eligible_actions = self.maze.eligible_actions(*self.maze.current_position)
        return self.rng.choice(eligible_actions)

    def learn(self, reward, state, action, state_prime, action_prime):
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
        self.Q = pd.DataFrame(
            index=[
                (i, j)
                for i in range(self.maze.maze.shape[0])
                for j in range(self.maze.maze.shape[1])
            ],
            columns=range(4),
            dtype=np.float,
        )
        self.Q.loc[:, :] = start_Q

        # set exits to 0
        exits = np.transpose(self.maze.get_exits())
        for x, y in exits:
            self.Q.loc[[(x, y)], :] = start_Q_for_exits

        # set impossible moves to some value
        # for idx, row in self.Q.iterrows():
        #     x, y = idx
        #     eligible_actions = self.maze.eligible_actions(x, y)
        #     for action in range(4):
        #         if action not in eligible_actions:
        #             row[action] = 0

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

        # decide whether to take a random action or not
        if self.rng.uniform() < self.epsilon:
            action = self.rng.choice(eligible_actions)
        else:
            action = (
                self.Q.loc[[self.maze.current_position], eligible_actions]
                .squeeze()
                .idxmax()
            )

        return action

    def learn(self, reward, state, action, state_prime, action_prime):
        """
        Update the agent's knowledge of the environment.

        Parameters
        ----------
        action : int
            The action taken.
        reward : float
            The reward received.
        """

        # get current and previous positions
        x_new, y_new = state_prime
        x_old, y_old = state

        QSA = self.Q.loc[[(x_new, y_new)], action_prime].values[0]
        QSA_old = self.Q.loc[[(x_old, y_old)], action].values[0]
        self.Q.loc[[(x_old, y_old)], action] += self.alpha * (
            reward + self.gamma * QSA - QSA_old
        )
