"""Module to run a maze game."""

from time import sleep
from turtle import clear

import numpy as np
from IPython.display import clear_output

from vizualiser import plot
from agent import RandomAgent
from maze import Maze


def run_game(seed=None):
    """
    Run a game.
    """
    maze = Maze((10, 10), 3, seed=seed)
    agent = RandomAgent(maze, seed=seed)
    done = False
    while not done:
        action = agent.act()
        reward, done = maze.move(action)
        agent.learn(action, reward)

        # plot the current step
        clear_output(wait=True)
        plot(maze)
        sleep(1)
