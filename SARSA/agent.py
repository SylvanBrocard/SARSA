"""
Agent class for the SARSA algorithm.
"""

from abc import ABC, abstractmethod
import numpy as np

class Agent(ABC):
    """Abstract agent class."""
    def __init__(self, maze, seed=None):
        """
        Initialize the agent.

        Parameters
        ----------
        maze : maze.Maze
            The environment to interact with.
        """
        self.maze = maze
        self.rng = np.random.default_rng(seed=seed)

    @abstractmethod
    def act(self):
        """
        Choose an action to take.

        Parameters
        ----------
        state : object
            The current state.

        Returns
        -------
        action : int
            The action to take.
        """
        pass

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
    def act(self):
        eligible_actions = self.maze.eligible_actions(*self.maze.current_position)
        return self.rng.choice(eligible_actions)

    def learn(self, action, reward):
        pass