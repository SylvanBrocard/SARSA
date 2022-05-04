"""Module to run a maze game."""

from time import sleep
from turtle import clear

import numpy as np
from IPython.display import clear_output
from tqdm import tqdm

from vizualiser import plot_maze
from agent import Agent
from maze import Maze

class Game():
    """
    Game class.
    """
    def __init__(self, maze:Maze, agent:Agent, max_steps=100) -> None:
        """
        Constructor.
        """
        self.maze = maze
        self.agent = agent
        self.rewards = []

    def run_game(self, plot:bool=False) -> None:
        """
        Run a game.
        """
        self.maze.current_position = self.maze.generate_start_position()
        if plot:
            plot_maze(self.maze)
            sleep(1)
        maze_done = False
        agent_done = False
        while not (maze_done or agent_done):
            action = self.agent.act()
            reward, maze_done = self.maze.step(action)
            agent_done = self.agent.max_steps + self.agent.reward <= 0
            self.agent.learn(action, reward)

            if plot:
                # plot the current step
                clear_output(wait=True)
                plot_maze(self.maze)
                sleep(1)

    def train_agent(self, episodes:int) -> None:
        """
        Train an agent.
        """
        for _ in tqdm(range(episodes), desc="episodes"):
            self.agent.reinit()
            self.run_game(plot=False)
            self.rewards.append(self.agent.reward)

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