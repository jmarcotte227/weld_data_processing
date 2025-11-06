import pickle
import matplotlib.pyplot as plt
import yaml
import sys
import numpy as np
from motoman_def import robot_obj, positioner_obj
from robotics_utils import H_inv
sys.path.append('../../Welding_Motoman/toolbox')
from angled_layers import avg_by_line, rotate, LiveFilter
import scienceplots

def rms_error(data):
    data = np.array(data)
    n = 0
    num = 0
    for i in data:
        if not np.isnan(i): 
            num = num + i**2
            n+=1
    return np.sqrt(num/n)

config_dir = "../../Welding_Motoman/config/"
dataset = "bent_tube/"
sliced_alg = "slice_ER_4043_large_hot/"
data_dir = "../../Welding_Motoman/data/" + dataset + sliced_alg

plt.style.use('science')
# colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# print('\n'.join(color for color in colors))
# exit()
plt.rcParams['text.usetex'] = True

fig,ax=plt.subplots()
fig.set_size_inches(3,2)
fig.set_dpi(300)
marker_size = 2
plt_colors = [
    '#0C5DA5',
    '#00B945',
    '#FF9500',
    '#FF2C00',
]
plt_styles = [
    'solid',
    'dotted',
    'dashed',
    'dashdot'
]
marker_styles = [
    'o',
    '^',
    's',
    'D'
]
layer_error = np.loadtxt("error_data/ER4043_bent_tube_large_hot_streaming_2025_03_06_feedback_troubleshooting_layer_err.csv", delimiter=',')

error = layer_error[101, :]
idx = np.linspace(1,len(error)+1, 49)
error_flat = np.zeros(len(error)*2)
idx_flat = np.zeros(len(error)*2) 
for i, err in enumerate(error):
    error_flat[2*i] = err
    error_flat[2*i+1] = err
    idx_flat[2*i] = idx[i]
    idx_flat[2*i+1] = idx[i+1]

ax.plot(idx_flat, error_flat, 'orange')
ax.spines[['right', 'top']].set_visible(False)
ax.axes.set_xticklabels([])
ax.axes.set_xticks([])
ax.axes.set_yticklabels([])
ax.axes.set_yticks([])
ax.set_xlabel("Path Index")
ax.set_ylabel("Error")
fig.savefig('ga_layer_error_19.png', dpi=fig.dpi)
plt.show()
    # if layer == 101:
    #     filter = LiveFilter()
    #     for i in range(flame.shape[0]):
    #         flames_filter.append(filter.process(flame[i,1:])[2])
    #     time = np.linspace(0,len(flame)/30, len(flame))
    #     start_idx = 32
    #     end_idx = -300
    #     print((time[-1]))
    #     ax.plot(time[start_idx:end_idx],flame[start_idx:end_idx,3], c=plt_colors[0], alpha=0.3)
    #     ax.plot(time[start_idx:end_idx],flames_filter[start_idx:end_idx], c=plt_colors[0])
        # ax.set_xlabel("Time")
        # ax.set_ylabel("Error")
    #     ax.set_xlim([0,30])
    #     # ax.set_title(f"Height Filter Comparison Layer {layer-82}")
    #     ax.legend(["Raw","Filtered"],
    #               facecolor='white', 
    #               framealpha=0.8,
    #               frameon=True,
    #    # loc='lower center',
    #    # ncol=2,
    #    # bbox_to_anchor=(0.5,-0.8)
    #    )
    #     # ax.grid()
    #     # ax.set_axis_off()
    #     ax.spines[['right', 'top']].set_visible(False)
    #     ax.axes.set_xticklabels([])
    #     ax.axes.set_xticks([])
    #     ax.axes.set_yticklabels([])
    #     ax.axes.set_yticks([])
    #     plt.show()

    # # plt.plot(flame[:,1], flame[:,3])
    # averages= avg_by_line(flame[:,0], flame[:,1:], np.linspace(0,49,50))
    # height_err.append(averages[:,2])
    # flames_flat.append(averages)
    # # if layer == 75:
    # #     plt.plot(averages[:,0], averages[:,2])
    # #     plt.show()
# rms_err = []
# for scan in height_err:
    # rms_err.append(rms_error(scan[1:-1]))
    # height_err_trim.append(scan[1:-1])
# rms_errs.append(rms_err)
# ax.set_aspect('equal')
# plt.show()

fig,ax = plt.subplots()

# np.savetxt(title+'_err.csv',rms_errs[0])
# np.savetxt(title+'_layer_err.csv',height_err_trim, delimiter=',')
