import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scienceplots

def zoom_outside(srcax, roi, dstax, color="red", linewidth=2, roiKwargs={}, arrowKwargs={}):
    '''Create a zoomed subplot outside the original subplot
    
    srcax: matplotlib.axes
        Source axis where locates the original chart
    dstax: matplotlib.axes
        Destination axis in which the zoomed chart will be plotted
    roi: list
        Region Of Interest is a rectangle defined by [xmin, ymin, xmax, ymax],
        all coordinates are expressed in the coordinate system of data
    roiKwargs: dict (optional)
        Properties for matplotlib.patches.Rectangle given by keywords
    arrowKwargs: dict (optional)
        Properties used to draw a FancyArrowPatch arrow in annotation
    '''
    roiKwargs = dict([("fill", False), ("linestyle", "dashed"),
                      ("color", color), ("linewidth", linewidth)]
                     + list(roiKwargs.items()))
    arrowKwargs = dict([("arrowstyle", "-"), ("color", color),
                        ("linewidth", linewidth)]
                       + list(arrowKwargs.items()))
    # draw a rectangle on original chart
    srcax.add_patch(Rectangle([roi[0], roi[1]], roi[2]-roi[0], roi[3]-roi[1], 
                            **roiKwargs))
    # get coordinates of corners
    srcCorners = [[roi[0], roi[1]], [roi[0], roi[3]],
                  [roi[2], roi[1]], [roi[2], roi[3]]]
    dstCorners = dstax.get_position().corners()
    srcBB = srcax.get_position()
    dstBB = dstax.get_position()
    # find corners to be linked
    if srcBB.max[0] <= dstBB.min[0]: # right side
        if srcBB.min[1] < dstBB.min[1]: # upper
            corners = [1, 2]
        elif srcBB.min[1] == dstBB.min[1]: # middle
            corners = [0, 1]
        else:
            corners = [0, 3] # lower
    elif srcBB.min[0] >= dstBB.max[0]: # left side
        if srcBB.min[1] < dstBB.min[1]:  # upper
           corners = [0, 3]
        elif srcBB.min[1] == dstBB.min[1]: # middle
            corners = [2, 3]
        else:
            corners = [1, 2]  # lower
    elif srcBB.min[0] == dstBB.min[0]: # top side or bottom side
        if srcBB.min[1] < dstBB.min[1]:  # upper
            corners = [0, 2]
        else:
            corners = [1, 3] # lower
    else:
        RuntimeWarning("Cannot find a proper way to link the original chart to "
                       "the zoomed chart! The lines between the region of "
                       "interest and the zoomed chart wiil not be plotted.")
        return
    # plot 2 lines to link the region of interest and the zoomed chart
    for k in range(2):
        srcax.annotate('', xy=srcCorners[corners[k]], xycoords="data",
            xytext=dstCorners[corners[k]], textcoords="figure fraction",
            arrowprops=arrowKwargs)

plt.style.use('science')
# colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# print('\n'.join(color for color in colors))
# exit()
plt.rcParams['text.usetex'] = True

fig, ax= plt.subplots()
fig.set_size_inches(5,4)
fig.set_dpi(300)
marker_size = 15
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
# labels = [
#         "Open-Loop Cold Model", 
#         "Open-Loop Hot Model",
#         "Closed-Loop Cold Model", 
#         "Closed-Loop Hot Model"
#         ]
labels = [
        'Layer-to-Layer\nCorrection Only',
        'Layer-to-Layer and\nIn-Layer Correction'
        ]

err_set = [
    # 'error_data/ER4043_bent_tube_large_cold_OL_2024_11_14_11_56_43_err.csv',
    # 'error_data/ER4043_bent_tube_large_hot_OL_2024_11_14_13_05_38_err.csv',
    # 'error_data/ER4043_bent_tube_large_cold_2024_11_07_10_21_39_err.csv',
    'error_data/ER4043_bent_tube_large_hot_2024_11_06_12_27_19_err.csv',
    # 'error_data/ER4043_bent_tube_2024_09_04_12_23_40_err.csv',
    'error_data/ER4043_bent_tube_large_hot_streaming_2025_03_06_feedback_troubleshooting_err.csv'
]
idx_start = 82
layer_no = np.linspace(1, 105-idx_start, 105-idx_start)
print(layer_no)
err_data=np.loadtxt(err_set[0], delimiter=',')
err_data = err_data[idx_start:-1]
ax.scatter(
        layer_no,
        err_data, 
        s=marker_size, 
        marker=marker_styles[0],
        label = labels[0],
        color = plt_colors[1]
        )
ax.plot(
        layer_no,
        err_data,
        alpha=0.3,
        color = plt_colors[1]
        )
print("-----Planning Stats------")
print("Mean Error: ", np.mean(err_data))
print("Peak Error: ", np.max(err_data))
print("RMSE Error: ", np.sqrt(np.mean((err_data)**2)))
err_data=np.loadtxt(err_set[1], delimiter=',')
err_data = err_data[idx_start:]
num_points = len(err_data)
ax.scatter(
        layer_no,
        err_data, 
        s=marker_size, 
        marker=marker_styles[1],
        label = labels[1],
        color = plt_colors[0]
        )
ax.plot(
        layer_no,
        err_data,
        alpha=0.3,
        color = plt_colors[0]
        )

# print stats
print("-----Correction Stats------")
print("Mean Error: ", np.mean(err_data))
print("Peak Error: ", np.max(err_data))
print("RMSE Error: ", np.sqrt(np.mean((err_data)**2)))

# err_data=np.loadtxt(err_set[-1], delimiter=',')
# ax1.plot(np.linspace(1,80,80), err_data)
# ax2.plot(np.linspace(1,80,80), err_data)

ax.set_xlabel("Layer Number")
ax.set_ylabel("RMSE (mm)")
ax.legend(facecolor='white', 
           framealpha=0.8,
           frameon=True,
           # loc='lower center',
           # ncol=2,
           # bbox_to_anchor=(0.5,-0.8)
           )
ax.grid()
# zoom_outside(ax1, [-5, -0.5, 111, 2], ax2, color='black', linewidth=0.5)
ax.set_title('Layer Error')
# ax2.set_title('Layer Error Zoomed')
# ax1.set_xlim(70,108)
# ax2.set_xlim(70,108)
fig.savefig('rms_plot_comp.png', dpi=fig.dpi)
plt.show()
