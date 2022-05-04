"""Module to run a maze game."""

from time import sleep

import numpy as np
from IPython.display import clear_output
from tqdm import tqdm

from agent import Agent
from maze import Maze
from vizualiser import plot_maze


class Game:
    """
    Game class.
    """

    def __init__(self, maze: Maze, agent: Agent, max_steps=100) -> None:
        """
        Constructor.

        Parameters
        ----------
        maze : maze.Maze
            The maze to play the game on.
        agent : agent.Agent
            The agent to play the game with.
        max_steps : int
            The maximum number of steps to take.
        """
        self.maze = maze
        self.agent = agent
        self.max_steps = max_steps
        self.rewards = []

    def run_game(self, plot: bool = False) -> None:
        """
        Run a game.

        Parameters
        ----------
        plot : bool
            Whether to plot the game.

        Returns
        -------
        reward : float
            The reward received.
        """
        self.maze.current_position = self.maze.generate_start_position()
        reward = 0
        steps = 0
        if plot:
            plot_maze(self.maze)
            sleep(1)
        maze_done = False
        agent_done = False
        action = self.agent.act()
        state = self.maze.current_position
        while not (maze_done or agent_done):
            maze_done = self.maze.step(action)
            if not maze_done:
                reward -= 1
            steps += 1
            agent_done = self.max_steps - steps <= 0
            action_prime = self.agent.act()
            state_prime = self.maze.current_position

            self.agent.learn(reward, state, action, state_prime, action_prime)

            state = state_prime
            action = action_prime

            if plot:
                # plot the current step
                clear_output(wait=True)
                plot_maze(self.maze)
                sleep(1)

        return reward

    def train_agent(self, episodes: int) -> None:
        """
        Train an agent.
        """
        for _ in tqdm(range(episodes), desc="episodes"):
            reward = self.run_game(plot=False)
            self.rewards.append(reward)

    def get_cumsum(self) -> np.ndarray:
        """
        Get the cumulative sum of rewards.
        """
        return np.cumsum(self.rewards)

    def get_cumav(self) -> np.ndarray:
        """
        Get the cumulative average of rewards.
        """
        return np.cumsum(self.rewards) / np.arange(1, len(self.rewards) + 1)
