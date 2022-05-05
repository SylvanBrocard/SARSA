from SARSA.maze import Maze, MazeWithGhosts
from SARSA.game import Game
from SARSA.agent import SARSAAgent

maze = Maze((10, 10), 3)
sarsaaagent = SARSAAgent(maze)
game = Game(maze, sarsaaagent, max_steps=20)
game.train_agent(episodes=1000)