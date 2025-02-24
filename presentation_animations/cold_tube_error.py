import pickle, sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import matplotlib.colors as mcolors

sys.path.append("../../Welding_Motoman/toolbox")
from angled_layers import avg_by_line

DATASET = "ER4043_bent_tube_large_cold_OL_2024_11_14_11_56_43"

DATA_DIR = "../../Welding_Motoman/scan/angled_layer/"
FLAME_DIR = DATA_DIR+"processing_data/"
ERROR_DIR = DATA_DIR+"error_data/"

PATH_DIR = "../../Welding_Motoman/data/bent_tube/slice_ER_4043_large_hot/curve_sliced_relative/"

# Load flame data
with open(FLAME_DIR+DATASET+"_flame.pkl","rb") as file:
    flames = pickle.load(file)
# Average and store in a single Array
job_no_offset = 3
flames_whole = []
for flame in flames:
    job_no = flame[:,0]-job_no_offset
    averages = avg_by_line(job_no, flame[:,1:], np.linspace(0,49,50))
    averages = averages[1:-1]
    for line in averages:
        flames_whole.append(line)


flames_whole = np.array(flames_whole)
print(len(flames_whole))

# Load error data
errors = np.loadtxt(ERROR_DIR+DATASET+"_err.csv", delimiter=',')
layer_errors = np.loadtxt(ERROR_DIR+DATASET+"_layer_err.csv", delimiter=',')
layer_errors = layer_errors.flatten()

# load path data
path_data = []
for i in range(107):
    path=np.loadtxt(PATH_DIR+f"slice{i}_0.csv", delimiter=',')[:,:3]
    for j in path:
        path_data.append(j)

path_data = np.array(path_data)
print(path_data.shape)

# Initialize figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')



num_points = flames_whole.shape[0]

# Additional values for coloring
colormap = cm.inferno  # Choose a colormap
colors = colormap(layer_errors)  # Map values to colors

# Initialize empty plot elements
lead_point, = ax.plot([], [], [], 'ro', markersize=8)  # Leading point
start = 5000
def update(frame):
    frame=frame+start
    ax.clear()
    ax.azim=200
    ax.set_ylim(-121, 40)
    ax.set_xlim(-51, 50)
    ax.set_zlim(-6, 200)
    ax.set_aspect('equal')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("2D Points Popping In with Color Trail")

    # plto nominal plan
    ax.scatter(path_data[:,0], path_data[:,1], path_data[:,2], alpha=0.002, color='r')
    
    # Plot colored line segments
    for i in range(frame):
        ax.plot(flames_whole[i:i+2, 0], flames_whole[i:i+2, 1], flames_whole[i:i+2, 2],
                color=colors[i], linewidth=2)
    
    # Update leading point
    ax.plot(flames_whole[frame, 0:1], flames_whole[frame, 1:2], flames_whole[frame, 2:3],
            'ro', markersize=8)
    
    return ax,

# Create animation
ani = FuncAnimation(fig, update, frames=num_points-start, interval=1, blit=False)
# ani.save('cold_ol.gif', writer='imagemagick')
plt.show()
