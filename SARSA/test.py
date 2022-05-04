from maze import Maze
from game import Game
from agent import SARSAAgent


# if __name__=="main":
print("test")
maze = Maze((5, 5), 3)
sarsaaagent = SARSAAgent(maze)
game = Game(maze, sarsaaagent)
game.train_agent(episodes=1000)