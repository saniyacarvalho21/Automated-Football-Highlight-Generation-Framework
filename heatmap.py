# heatmap.py
import matplotlib.pyplot as plt

def heat(points):
    x,y=zip(*points)
    plt.hist2d(x,y,bins=30)
    plt.savefig("data/outputs/heatmap.png")