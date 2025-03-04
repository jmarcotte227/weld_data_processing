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

DATASET = "ER4043_bent_tube_large_hot_2024_11_06_12_27_19"

DATA_DIR = "../../Welding_Motoman/scan/angled_layer/"
FLAME_DIR = DATA_DIR+"processing_data/"
ERROR_DIR = DATA_DIR+"error_data/"
TEMP_DIR = DATA_DIR+"temp_data/"

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
print(flames_whole.shape)

# Temp data
with open(TEMP_DIR+DATASET+"_temps.pkl","rb") as file:
    temps = pickle.load(file)
# Average and store in a single Array
job_no_offset = 3
temps_whole = []
for flame in temps:
    job_no = flame[:,0]-job_no_offset
    averages = avg_by_line(job_no, flame[:,1], np.linspace(0,49,50))
    averages = averages[1:-1]
    for line in averages:
        temps_whole.append(line)


temps_whole = np.array(temps_whole)
temps_whole = temps_whole.flatten()
print(temps_whole.shape)

# Load error data
errors = np.loadtxt(ERROR_DIR+DATASET+"_err.csv", delimiter=',')
layer_errors = np.loadtxt(ERROR_DIR+DATASET+"_layer_err.csv", delimiter=',')
layer_errors = -1*layer_errors.flatten()

# load path data
path_data = []
for i in range(107):
    path=np.loadtxt(PATH_DIR+f"slice{i}_0.csv", delimiter=',')[:,:3]
    for j in path:
        path_data.append(j)

path_data = np.array(path_data)

# def animate(num):
#     print(num)
#     data = flames_whole[:num, :]
#     art.set_offsets(data)
#     art.set_color(cmap(norm(layer_errors[:num])))
#     return art,

def update(num):
    print(num)
    # line.set_data_3d(flames_whole[:num,:].T)
    line._offsets3d=(flames_whole[:num,0],flames_whole[:num,1],flames_whole[:num,2])
    line.set_color(colors[:num])
    # point.set_data(flames_whole[num,0], flames_whole[num,1])
    # point.set_3d_properties(flames_whole[num,2], 'z')
    point._offsets3d=(flames_whole[num:num+1,0],flames_whole[num:num+1,1],flames_whole[num:num+1,2])
    point.set_color('r')
    return point,

# Initialize figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# norm = matplotlib.colors.Normalize(vmin=0.1870684015219398, vmax=52.71549799054008)
# colors = cm.jet(norm(layer_errors))
norm = matplotlib.colors.Normalize(vmin=np.nanmin(temps_whole), vmax=np.nanmax(temps_whole))
colors = cm.jet(norm(temps_whole))
print("min: ", np.nanmin(temps_whole))
print("max: ", np.nanmax(temps_whole))


line = ax.scatter([],[],[],facecolors=[], marker='o')
# line = ax.plot([],[],[],)[0]
# point = ax.plot([],[],[], marker='o',color='r')[0]
point = ax.scatter([],[],[],facecolors=[], marker='o', s=20)
# ax.scatter(path_data[:,0],path_data[:,1],path_data[:,2], alpha=0.01, c='r')
s_m = matplotlib.cm.ScalarMappable(cmap=cm.jet, norm=norm)
s_m.set_array([])


ax.azim=200
ax.set_ylim(-121, 40)
ax.set_xlim(-51, 50)
ax.set_zlim(-6, 200)
ax.set_aspect('equal')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
cbar = fig.colorbar(s_m, ax=ax, label="Error (mm)")
cbar.set_label("Brightness Temperature", size=18)
# cbar.set_label("Error (mm)", size=18)

# art = ax.scatter([],[],c=[])

# ani = FuncAnimation(fig, animate, frames = 100, interval=5, blit=True)
# ani = FuncAnimation(fig, update, frames = flames_whole.shape[0], interval=5, repeat=False)
ani = FuncAnimation(fig, update, frames = 5087, interval=5, repeat=False)

# Create animation
# ani.save('hot_cl_temp.mp4', dpi=300)
plt.show()
