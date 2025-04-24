import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button

stop_flag = False
start_flag = True  

def stop_plot(event):
    global stop_flag, start_flag
    stop_flag = True
    start_flag = False

def start_plot(event):
    global stop_flag, start_flag
    stop_flag = False
    start_flag = True

plt.ion()

fig = None
i = 0 

while i < 100:
    if stop_flag:
        plt.pause(0.1)
        continue

    x = np.linspace(0, 10, 100)
    y1 = np.sin(x + i * 0.1) + np.random.normal(0, 0.1, size=100)
    y2 = np.cos(x + i * 0.1) + np.random.normal(0, 0.1, size=100)
    y3 = np.tan(x + i * 0.05) + np.random.normal(0, 0.5, size=100)

    x3d, y3d, z3d = np.random.rand(3, 100)
    heat = np.random.rand(10, 10)

    if fig is None:
        fig = plt.figure(figsize=(18, 10))
        ax1 = fig.add_subplot(2, 3, 1)
        ax2 = fig.add_subplot(2, 3, 2, projection='3d')
        ax3 = fig.add_subplot(2, 3, 3)
        ax4 = fig.add_subplot(2, 3, 4)
        ax5 = fig.add_subplot(2, 3, 5)
        ax6 = fig.add_subplot(2, 3, 6)

        ax_stop = fig.add_axes([0.40, 0.92, 0.1, 0.05])
        ax_start = fig.add_axes([0.52, 0.92, 0.1, 0.05])
        btn_stop = Button(ax_stop, 'Stop')
        btn_start = Button(ax_start, 'Start')
        btn_stop.on_clicked(stop_plot)
        btn_start.on_clicked(start_plot)

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.cla()

    ax1.plot(x, y1, label='sin + noise')
    ax1.set_title("RangeFFT")
    ax1.legend()

    ax2.scatter(x3d, y3d, z3d, c=z3d, cmap='viridis')
    ax2.set_title("PointCloud")

    sns.heatmap(heat, ax=ax3, cbar=False, cmap='coolwarm')
    ax3.set_title("Range-doppler Heatmap")

    ax4.plot(x, y2, color='orange')
    ax4.set_title("Tracked Phase")
    ax4.legend()

    ax5.plot(x, y3, color='green')
    ax5.set_title("Tracked Phase Value")
    ax5.legend()

    ax6.axis('off')
    message = f"Iteration {i+1}/100\nMax Heat: {heat.max():.2f}\nMean sin: {y1.mean():.2f}"
    ax6.text(0.5, 0.5, message, ha='center', va='center', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.pause(0.1)

    i += 1  

plt.ioff()
plt.show()
