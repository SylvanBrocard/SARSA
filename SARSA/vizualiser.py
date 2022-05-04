import matplotlib.pyplot as plt
import numpy as np

from maze import Maze

def plot_maze():
    plt.figure(1, figsize=(12,10), dpi=100)
    plt.axis([0,1,0,1])
    plt.title("The Amazing Maze")
    plt.yticks(np.arange(0,y_shape+1,1))
    plt.xticks(np.arange(0,x_shape+1,1))
    plt.grid(color='r', linestyle='-', linewidth=1)
    plt.scatter(x=[0.5,3.5], y=[0.5,3.5], s = 130, c = 'yellow', marker = '*', edgecolors = 'green')

def plot_maze(maze:Maze):
    x_shape, y_shape = maze.shape
    exits_x, exits_y = maze.get_exits()
    agent_x, agent_y = maze.current_position

    plt.figure(1, figsize=(12,10))
    plt.axis([0,1,0,1])
    plt.yticks(np.arange(0,y_shape+1,1))
    plt.xticks(np.arange(0,x_shape+1,1))
    plt.grid(color='r', linestyle='-', linewidth=1)

    # add exits
    plt.scatter(x=exits_x+0.5, y=exits_y+0.5, s = 130, c = 'yellow', marker = '*', edgecolors = 'green')

    # add agent
    plt.scatter(x=agent_x+0.5, y=agent_y+0.5, s = 130, c = 'black', marker = 'o', edgecolors = 'blue')

    # plt.imshow(maze, cmap='gray', interpolation='nearest')
    plt.show()

if __name__ == "__main__":
    matrice_jeu = [[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]]
    matrice_jeu = np.array(matrice_jeu)
    x_shape,y_shape = matrice_jeu.shape

    plot_maze()
    plt.show()