"""
Agent class for the SARSA algorithm.
"""

from abc import ABC, abstractmethod
from math import isnan

import numpy as np
import pandas as pd

from maze import Maze


class Agent(ABC):
    """Abstract agent class."""

    def __init__(self, maze: Maze, max_steps=100, seed=None):
        """
        Initialize the agent.

        Parameters
        ----------
        maze : maze.Maze
            The environment to interact with.
        """
        self.maze = maze
        self.rng = np.random.default_rng(seed=seed)
        self.max_steps = max_steps
        self.reward = 0

    def reinit(self):
        """
        Reinitialize the agent.
        """
        self.reward = 0

    @abstractmethod
    def choose_action(self):
        """
        Choose an action.
        """
        pass

    def act(self):
        """
        Choose an action to take.

        Returns
        -------
        action : int
            The action to take.
        """
        action = self.choose_action()
        self.reward -= 1
        return action

    @abstractmethod
    def learn(self, action, reward):
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

    def choose_action(self):
        eligible_actions = self.maze.eligible_actions(*self.maze.current_position)
        return self.rng.choice(eligible_actions)

    def learn(self, action, reward):
        pass


class SARSAAgent(Agent):
    """SARSA agent."""

    def __init__(
        self, maze: Maze, alpha=0.1, gamma=0.9, epsilon=0.1, max_steps=100, seed=None
    ):
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
        max_steps : int
            The maximum number of steps to take.
        """
        super().__init__(maze, max_steps, seed)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = None
        self.initialize_Q()
        self.old_position = None

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
        self.Q.loc[:, :] = 0
        # print(self.Q)
        # print(self.Q.index)
        # set exits to 0
        exits = np.transpose(self.maze.get_exits())
        for x, y in exits:
            self.Q.loc[[(x, y)], :] = 0

        # print(self.Q.index)
        # set impossible moves to negative infinity
        # for idx, row in self.Q.iterrows():
        #     x, y = idx
        #     eligible_actions = self.maze.eligible_actions(x, y)
        #     for action in range(4):
        #         if action not in eligible_actions:
        #             row[action] = 0

    def choose_action(self):
        """
        Choose an action with e-greedy.
        """
        eligible_actions = self.maze.eligible_actions(*self.maze.current_position)
        if self.rng.uniform() < self.epsilon:
            action = self.rng.choice(eligible_actions)
        else:
            # print(self.maze.current_position)
            action = self.Q.loc[[self.maze.current_position], eligible_actions].squeeze().idxmax()
            if action not in eligible_actions:
                raise ValueError(f"action {action} not in eligible actions\n{self.Q.loc[[self.maze.current_position]].squeeze()}")
        if isnan(action):
            raise ValueError(f"NaN action, with {self.rng.uniform() < self.epsilon}")
        self.old_position = self.maze.current_position
        # print(f"action= {action}")
        return action

    def learn(self, action, reward):
        x, y = self.maze.current_position
        x_old, y_old = self.old_position
        new_action = self.choose_action()
        # print(f"x,y,new action= {(x,y, new_action)}  ; x_old, y_old, action= {(x_old,y_old, action)}")
        QSA = self.Q.loc[[(x, y)], new_action].values[0]
        QSA_old = self.Q.loc[[(x_old, y_old)], action].values[0]
        self.Q.loc[[(x_old, y_old)], action] += self.alpha * (
            reward
            + self.gamma * QSA
            - QSA_old
        )
        if self.Q.loc[[(x_old, y_old)], action].isnull().values.any():
            raise ValueError(f"NaN Q value, with {self.Q.loc[[(x_old, y_old)], action]}")

