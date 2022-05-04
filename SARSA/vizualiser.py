import matplotlib.pyplot as plt
import numpy as np

matrice_jeu = [[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]]
matrice_jeu = np.array(matrice_jeu)
x_shape,y_shape = matrice_jeu.shape

def plot():
    plt.figure(1, figsize=(12,10), dpi=100)
    plt.axis([0,1,0,1])
    plt.title("The Amazing Maze")
    plt.yticks(np.arange(0,y_shape+1,1))
    plt.xticks(np.arange(0,x_shape+1,1))
    plt.grid(color='r', linestyle='-', linewidth=1)
    plt.scatter(x=[0.5,3.5], y=[0.5,3.5], s = 130, c = 'yellow', marker = '*', edgecolors = 'green')