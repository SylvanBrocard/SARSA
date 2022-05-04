"""Module to run a maze game."""

from time import sleep
from turtle import clear

import numpy as np
from IPython.display import clear_output

from vizualiser import plot_maze
from agent import RandomAgent, Agent
from maze import Maze


def run_game(maze:Maze, agent:Agent) -> None:
    """
    Run a game.
    """
    maze.current_position = maze.generate_start_position()
    plot_maze(maze)
    sleep(1)
    done = False
    while not done:
        action = agent.act()
        reward, done = maze.step(action)
        agent.learn(action, reward)

        # plot the current step
        clear_output(wait=True)
        plot_maze(maze)
        sleep(1)
