"""Module to run a maze game."""

from time import sleep

import numpy as np
from IPython.display import clear_output
from tqdm import tqdm

from .agent import Agent
from .maze import Maze
from .vizualiser import plot_maze


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

        # choisir une action a depuis s en utilisant la politique spécifiée par Q (par exemple ε-greedy)
        action = self.agent.act()
        # initialiser l'état s
        state = self.maze.current_position

        # répéter jusqu'à ce que s soit l'état terminal 
        maze_done = False
        agent_done = False
        while not (maze_done or agent_done):
            # exécuter l'action a
            maze_done = self.maze.step(action)

            # observer la récompense r et l'état s'
            if not maze_done:
                reward -= 1
            steps += 1
            state_prime = self.maze.current_position
            agent_done = self.max_steps - steps <= 0

            # choisir une action a' depuis s' en utilisant la politique spécifiée par Q (par exemple ε-greedy)
            action_prime = self.agent.act()

            # Q[s, a] := Q[s, a] + α[r + γQ(s', a') - Q(s, a)]
            self.agent.learn(reward, state, action, state_prime, action_prime)

            # s ← s'
            # a ← a'
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

class GameWithGhosts(Game):
    """
    Game class with ghosts.
    """

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
        self.maze.generate_ghosts()
        reward = 0
        steps = 0
        if plot:
            plot_maze(self.maze)
            sleep(1)

        # choisir une action a depuis s en utilisant la politique spécifiée par Q (par exemple ε-greedy)
        action = self.agent.act()
        # initialiser l'état s
        state = self.maze.get_state()

        # répéter jusqu'à ce que s soit l'état terminal 
        maze_done = False
        agent_done = False
        agent_dead = False
        while not (maze_done or agent_done or agent_dead):
            # exécuter l'action a
            maze_done, agent_dead = self.maze.step(action)

            # observer la récompense r et l'état s'
            if not maze_done:
                reward -= 1
            if agent_dead:
                reward -= 10
            steps += 1
            state_prime = self.maze.get_state()
            agent_done = self.max_steps - steps <= 0

            # choisir une action a' depuis s' en utilisant la politique spécifiée par Q (par exemple ε-greedy)
            action_prime = self.agent.act()

            # if no possible move, agent is dead
            if action_prime == -1:
                agent_dead = True
                reward -= 10
                break

            # Q[s, a] := Q[s, a] + α[r + γQ(s', a') - Q(s, a)]
            self.agent.learn(reward, state, action, state_prime, action_prime)

            # s ← s'
            # a ← a'
            state = state_prime
            action = action_prime

            if plot:
                # plot the current step
                clear_output(wait=True)
                plot_maze(self.maze)
                sleep(1)

        return reward