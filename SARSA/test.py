from maze import Maze
from game import Game
from agent import SARSAAgent
from vizualiser import plot_cumav

print("test")
maze = Maze((5, 5), 3)
sarsaaagent = SARSAAgent(maze)
game = Game(maze, sarsaaagent)
game.train_agent(episodes=100)

plot_cumav(game)