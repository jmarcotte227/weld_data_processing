import pickle, sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import matplotlib.colors as mcolors
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cycler import cycler

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

# Define your list of hex codes or color names
# my_colors = [
#     '#4477aa',
#     '#228833',
#     '#66ccee',
#     '#ccbb44',
#     '#ee6677',
#     '#aa3377',
# ]
my_colors = [
    '#009e73',
    '#0072b2',
    '#f0e442',
    '#e69f00',
    '#d55e00',
    '#cc79a7',
]

# Set the global color cycle
plt.rc('axes', prop_cycle=cycler(color=my_colors))

sys.path.append("../../../Welding_Motoman/toolbox")
from angled_layers import avg_by_line

# DATASET = "ER4043_bent_tube_large_cold_OL_2024_11_14_11_56_43"
DATASET = "ER4043_bent_tube_large_hot_2024_11_06_12_27_19"

DATA_DIR = "../../../Welding_Motoman/scan/angled_layer/"
FLAME_DIR = DATA_DIR+"processing_data/"
ERROR_DIR = DATA_DIR+"error_data/"
TEMP_DIR = DATA_DIR+"temp_data/"

PATH_DIR = "../../Welding_Motoman/data/bent_tube/slice_ER_4043_large_hot/curve_sliced_relative/"

fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})

# Load flame data
with open(FLAME_DIR+DATASET+"_flame.pkl","rb") as file:
    flames = pickle.load(file)
# Average and store in a single Array
job_no_offset = 3
flames_whole = []
for flame in flames:
    job_no = flame[:,0]-job_no_offset
    averages = avg_by_line(job_no, flame[:,1:], np.linspace(0,49,50))
    averages = averages[1:]
    ax.plot(averages[:,1], averages[:,0], averages[:,2])
    for line in averages:
        flames_whole.append(line)


ax.set_aspect('equal')
ax.set_xlabel("$X$")
ax.set_ylabel("$Y$")
ax.set_zlabel("$Z$")
ax.locator_params(axis='y', nbins=3)
ax.locator_params(axis='x', nbins=5)
ax.view_init(elev=10., azim=150)
fig.set_size_inches(3.5, 5.5)
fig.subplots_adjust(
    left=-0.2,
    bottom=0,
    right=1.1,
    top=1.1,
    wspace=0,
    hspace=0
)
plt.savefig(f"output_plots/exp_tube_vis.png", dpi=300)
plt.savefig(f"output_plots/exp_tube_vis.tiff", dpi=300)
plt.show()
