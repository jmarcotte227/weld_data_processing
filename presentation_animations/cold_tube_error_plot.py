import pickle, sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append("../../Welding_Motoman/toolbox")
from angled_layers import avg_by_line

DATASET = "ER4043_bent_tube_large_cold_OL_2024_11_14_11_56_43"

DATA_DIR = "../../Welding_Motoman/scan/angled_layer/"
FLAME_DIR = DATA_DIR+"processing_data/"
ERROR_DIR = DATA_DIR+"error_data/"

PATH_DIR = "../../Welding_Motoman/data/bent_tube/slice_ER_4043_large_hot/curve_sliced_relative/"

# # Load flame data
# with open(FLAME_DIR+DATASET+"_flame.pkl","rb") as file:
#     flames = pickle.load(file)
# # Average and store in a single Array
# job_no_offset = 3
# flames_whole = []
# for flame in flames:
#     job_no = flame[:,0]-job_no_offset
#     averages = avg_by_line(job_no, flame[:,1:], np.linspace(0,49,50))
#     averages = averages[1:-1]
#     for line in averages:
#         flames_whole.append(line)


# flames_whole = np.array(flames_whole)

# Load error data
errors = np.loadtxt(ERROR_DIR+DATASET+"_err.csv", delimiter=',')
layers = np.linspace(1,len(errors), len(errors))
print(layers)
# layer_errors = np.loadtxt(ERROR_DIR+DATASET+"_layer_err.csv", delimiter=',')
# layer_errors = layer_errors.flatten()

# load path data
# path_data = []
# for i in range(107):
#     path=np.loadtxt(PATH_DIR+f"slice{i}_0.csv", delimiter=',')[:,:3]
#     for j in path:
#         path_data.append(j)

# path_data = np.array(path_data)

# def animate(num):
#     print(num)
#     data = flames_whole[:num, :]
#     art.set_offsets(data)
#     art.set_color(cmap(norm(layer_errors[:num])))
#     return art,

def update(num):
    print(num)
    line.set_data(layers[:num],errors[:num])
    # line._offsets3d=(flames_whole[:num,0],flames_whole[:num,1],flames_whole[:num,2])
    # line.set_color(colors[:num])
    # point.set_data(flames_whole[num,0], flames_whole[num,1])
    # point.set_3d_properties(flames_whole[num,2], 'z')
    point.set_offsets([layers[num],errors[num]])
    return point,

# Initialize figure and 3D axis
fig = plt.figure()
fig.set_size_inches(8,3)
ax = fig.add_subplot(111)

line = ax.plot([],[],[],c='b')[0]
# line = ax.plot([],[],[],)[0]
# point = ax.plot([],[],[], marker='o',color='r')[0]
point = ax.scatter([],[],c='b' ,marker='o', s=20)


ax.grid()
ax.set_xlim([-1, 107])
ax.set_ylim([-5, 50])
ax.set_xlabel("Layer Number", fontsize=18)
ax.set_ylabel("RMSE (mm)", fontsize=18)
fig.tight_layout()
# art = ax.scatter([],[],c=[])

# ani = FuncAnimation(fig, animate, frames = 100, interval=5, blit=True)
# ani = FuncAnimation(fig, update, frames = flames_whole.shape[0], interval=5, repeat=False)
ani = FuncAnimation(fig, update, frames = 106, interval=50, repeat=False)

# Create animation
ani.save('cold_ol_plot.mp4', dpi=300)
plt.show()
