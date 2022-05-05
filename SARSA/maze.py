"""
Module to generate a maze.
"""
from typing import Tuple
from copy import deepcopy

import numpy as np


class Maze:
    """
    Maze class.
    """

    # list of possible actions, corresponding to left, right, up, down
    actions = [0, 1, 2, 3]

    def __init__(self, shape: Tuple[int, int], exits: int, seed: int = None):
        """
        Constructor.

        Parameters
        ----------
        shape : Tuple[int,int]
            Shape of the maze.
        exits : int
            Number of exits.
        seed : int
            Seed for the random number generator.
        """
        self.shape = shape
        self.exits = exits
        self.rng = np.random.default_rng(seed=seed)
        self.maze = None
        self.generate_maze()
        self.current_position = self.generate_start_position()

    def generate_maze(self) -> np.ndarray:
        """
        Generate a maze.
        """
        self.maze = np.zeros(self.shape, dtype=np.int)
        for _ in range(self.exits):
            x, y = self.generate_exit()
            self.maze[x, y] = 1

    def generate_exit(self) -> Tuple[int, int]:
        """
        Generate an exit.

        Returns
        -------
        x : int
            X coordinate.
        y : int
            Y coordinate.
        """
        x, y = self.generate_random_coord()
        while self.maze[x, y] == 1:
            x, y = self.generate_random_coord()
        return x, y

    def generate_start_position(self) -> Tuple[int, int]:
        """
        Generate a start position.

        Returns
        -------
        x : int
            X coordinate.
        y : int
            Y coordinate.
        """
        x, y = self.generate_random_coord()
        while self.maze[x, y] == 1:
            x, y = self.generate_random_coord()
        return x, y

    def generate_random_coord(self) -> Tuple[int, int]:
        """
        Generate random coordinates.

        Returns
        -------
        x : int
            X coordinate.
        y : int
            Y coordinate.
        """
        x = self.rng.integers(self.shape[0])
        y = self.rng.integers(self.shape[1])
        return x, y

    def valid_coordinates(self, x: int, y: int) -> bool:
        """
        Check if the coordinates are valid.

        Parameters
        ----------
        x : int
            X coordinate.
        y : int
            Y coordinate.

        Returns
        -------
        valid : bool
            Whether the coordinates are valid.
        """
        return 0 <= x < self.shape[0] and 0 <= y < self.shape[1]

    def eligible_actions(self, x: int, y: int) -> list:
        """
        Return eligible moves.

        Parameters
        ----------
        x : int
            X coordinate.
        y : int
            Y coordinate.

        Returns
        -------
        eligible_actions : list
            List of eligible moves.
        """
        moves = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]
        return [i for i, (x, y) in enumerate(moves) if self.valid_coordinates(x, y)]

    def move(self, x: int, y: int, action: int) -> Tuple[int, int]:
        """
        Move in the maze.

        Parameters
        ----------
        x : int
            X coordinate.
        y : int
            Y coordinate.
        action : int
            Action to take.

        Returns
        -------
        x : int
            New X coordinate.
        y : int
            New Y coordinate.
        """
        if action == 0:
            y -= 1
        elif action == 1:
            y += 1
        elif action == 2:
            x -= 1
        elif action == 3:
            x += 1
        return x, y

    def step(self, action: int) -> bool:
        """
        Step in the maze.

        Parameters
        ----------
        action : int
            Action to take.

        Returns
        -------
        done : bool
            Whether the game is done.
        dead : bool
            Whether the player is dead. Always False.
        """
        x, y = self.current_position
        x, y = self.move(x, y, action)
        self.current_position = x, y
        done = self.maze[x, y] == 1
        return done, False

    def get_exits(self) -> list:
        """
        Return exits.

        Returns
        -------
        exits : list
            List of exits.
        """
        return np.where(self.maze == 1)

    def get_state(self) -> Tuple:
        """
        Get current world state.

        Returns
        -------
        player : Tuple[int,int]
            Player position.
        ghosts : list
            List of ghost positions. Always empty.
        """

        # get current position
        x, y = self.current_position
        player = (x, y)

        return player, []


class MazeWithGhosts(Maze):
    """
    Maze with ghosts.
    """

    def __init__(
        self, shape: Tuple[int, int], exits: int, nb_ghosts: int, seed: int = None
    ):
        """
        Constructor.

        Parameters
        ----------
        shape : Tuple[int,int]
            Shape of the maze.
        exits : int
            Number of exits.
        ghosts : int
            Number of ghosts.
        seed : int
            Seed for the random number generator.
        """
        super().__init__(shape, exits, seed)
        self.nb_ghosts = nb_ghosts
        self.ghosts = None
        self.generate_ghosts()

    def generate_ghosts(self):
        """
        Generate ghosts.
        """
        self.ghosts = []
        for _ in range(self.nb_ghosts):
            x, y = self.generate_ghost()
            self.ghosts.append([x, y])

    def generate_ghost(self) -> Tuple[int, int]:
        """
        Generate a ghost.

        Returns
        -------
        x : int
            X coordinate.
        y : int
            Y coordinate.
        """
        x, y = self.generate_random_coord()
        while [x, y] in self.ghosts or self.current_position == [x, y]:
            x, y = self.generate_random_coord()
        return x, y

    def eligible_actions(self, x: int, y: int) -> list:
        """
        Return eligible moves.

        Parameters
        ----------
        x : int
            X coordinate.
        y : int
            Y coordinate.

        Returns
        -------
        eligible_actions : list
            List of eligible moves.
        """
        moves = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]
        return [
            i
            for i, (x, y) in enumerate(moves)
            if self.valid_coordinates(x, y) and [x, y] not in self.ghosts
        ]

    def move_ghost(self, x: int, y: int):
        """
        Move ghost.

        Returns
        -------
        x : int
            X coordinate.
        y : int
            Y coordinate.
        """
        eligible_actions = self.eligible_actions(x, y)
        if len(eligible_actions) > 0:
            action = self.rng.choice(eligible_actions)
            x, y = self.move(x, y, action)
        return x, y

    def step(self, action: int) -> bool:
        """
        Step in the maze.

        Parameters
        ----------
        action : int
            Action to take.

        Returns
        -------
        done : bool
            Whether the game is done.
        dead : bool
            Whether the player is dead.
        """
        # move player
        x, y = self.current_position
        x, y = self.move(x, y, action)
        self.current_position = x, y

        # move ghosts
        for ghost in self.ghosts:
            x, y = ghost
            x, y = self.move_ghost(x, y)
            ghost[0], ghost[1] = x, y

        x, y = self.current_position
        dead = [x, y] in self.ghosts
        done = (self.maze[x, y] == 1 or dead)
        return done, dead

    def get_state(self) -> Tuple:
        """
        Get current world state.

        Returns
        -------
        player : Tuple[int,int]
            Player position.
        ghosts : list[list[int,int]]
            List of ghost positions.
        """

        # get current position
        x, y = self.current_position
        player = [x, y]

        # get ghosts
        ghosts = deepcopy(self.ghosts)

        state = player
        for ghost in ghosts:
            state.extend(ghost)
        return state