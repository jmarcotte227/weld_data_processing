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

plt.rcParams['text.usetex']=True

DATASETS =[
        "ER4043_bent_tube_large_cold_OL_2024_11_14_11_56_43",
        "ER4043_bent_tube_large_hot_2024_11_06_12_27_19"
]

DATA_DIR = "../../Welding_Motoman/scan/angled_layer/"
FLAME_DIR = DATA_DIR+"processing_data/"
ERROR_DIR = DATA_DIR+"error_data/"
TEMP_DIR = DATA_DIR+"temp_data/"

PATH_DIR = "../../Welding_Motoman/data/bent_tube/slice_ER_4043_large_hot/curve_sliced_relative/"

fig = plt.figure(figsize=(4.5,5))
ax = fig.subplots(2,2, sharey=True, sharex=True)
fig.set_dpi(300)

for idx, dataset in enumerate(DATASETS):
    # Temp data
    with open(TEMP_DIR+dataset+"_temps.pkl","rb") as file:
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
    # errors
    errors = np.loadtxt(ERROR_DIR+dataset+"_err.csv", delimiter=',')
    layers = np.linspace(1,len(errors), len(errors))
    layer_errors = np.loadtxt(ERROR_DIR+dataset+"_layer_err.csv", delimiter=',')
    layer_errors = -1*layer_errors.flatten()
    # Load flame data
    with open(FLAME_DIR+dataset+"_flame.pkl","rb") as file:
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
    norm = matplotlib.colors.Normalize(vmin=10193.589930555554, vmax=17227.5335)
    colors_temp = cm.jet(norm(temps_whole))
    norm_error = matplotlib.colors.Normalize(vmin=-2.4927526, vmax=52.715498)
    colors_error = cm.jet(norm_error(layer_errors))
    print(np.nanmin(layer_errors))
    print(np.nanmax(layer_errors))

    ax[0,idx].scatter(-flames_whole[:,1], flames_whole[:,2], c=colors_error, s=2)
    ax[1,idx].scatter(-flames_whole[:,1], flames_whole[:,2], c=colors_temp, s=2)
    ax[0,idx].set_aspect('equal')
    ax[1,idx].set_aspect('equal')


s_m = matplotlib.cm.ScalarMappable(cmap=cm.jet, norm=norm)
s_m.set_array([])
s_m_err = matplotlib.cm.ScalarMappable(cmap=cm.jet, norm=norm_error)
s_m_err.set_array([])
cbar = fig.colorbar(s_m_err, ax=ax[0,1], label="Error (mm)")
cbar = fig.colorbar(s_m, ax=ax[1,1], label="Brightness\nTemperature")
ax[0,0].set_title("Open-Loop")
ax[0,1].set_title("Closed-Loop")
fig.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
# fig.savefig("error_temp_plot.png", dpi=fig.dpi)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(flames_whole[:,0], -flames_whole[:,1], flames_whole[:,2], c=colors_temp, s=2)
ax.set_aspect('equal')
plt.show()
