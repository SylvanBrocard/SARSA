"""
Module to generate a maze.
"""

class Maze():
    """
    Maze class.
    """

    def __init__(self, shape):
        """
        Constructor.
        """
        self.shape = shape
        self.maze = self.generate_maze()