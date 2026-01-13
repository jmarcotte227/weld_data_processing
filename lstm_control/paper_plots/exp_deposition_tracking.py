import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns


rc('text', usetex=True)
rc('font',**{'family':'sans-serif','sans-serif':['Latin Modern Sans']})
# colors = ["#7fc97f","#beaed4","#fdc086"]
colors = [
    '#e69f00',
    '#009e73',
]
def rms(x):
    return np.sqrt(np.sum(np.square(x))/len(x))

DATA_DIR = "data/"
DATA_DIR = "../../../recorded_data/"
PROC_DATA_DIR = "../data/"
# load baseline test set

##### WITH NOISE #####
# baseline
DATA_1 = "2026_01_12_11_48_35_wall_lstm_baseline_control"
DATA_2 = "2026_01_12_10_21_38_wall_lstm_control"

LAYER = 20
dh_1 = np.loadtxt(f"{PROC_DATA_DIR}calc_dh/{DATA_1}_dh.csv", delimiter=',') [LAYER]
dh_d_1 = np.loadtxt(f"{DATA_DIR}{DATA_1}/layer_{LAYER}/dh_d.csv", delimiter=',')
u_cmd_1 = np.loadtxt(f"{PROC_DATA_DIR}v_set/{DATA_1}_v_cmd.csv", delimiter=',')[LAYER]

dh_2 = np.loadtxt(f"{PROC_DATA_DIR}calc_dh/{DATA_2}_dh.csv", delimiter=',') [LAYER]
dh_d_2 = np.loadtxt(f"{DATA_DIR}{DATA_2}/layer_{LAYER}/dh_d.csv", delimiter=',')
u_cmd_2 = np.loadtxt(f"{PROC_DATA_DIR}v_set/{DATA_2}_v_cmd.csv", delimiter=',')[LAYER]


ylim = [
    min([
        np.min(dh_1),
        np.min(dh_d_1),
        np.min(dh_2),
        np.min(dh_d_2)
    ])-0.1,
    max([
        np.max(dh_1),
        np.max(dh_d_1),
        np.max(dh_2),
        np.max(dh_d_2)
    ])+0.1
]

# Log Log Plots
fig, ax = plt.subplots(4,1, sharex=True)
ax[0].plot(
    dh_1,
    color=colors[0],
    label=f"Baseline Control $\\mathbf{{\\Delta H_{{{LAYER}}}}}$"
)
ax[0].plot(
    dh_d_1,
    color=colors[0],
    linestyle='dotted',
    label=f"Baseline Control $\\mathbf{{\\Delta H_{{d,{LAYER}}}}}$"
)
ax[1].plot(
    u_cmd_1,
    color=colors[0],
    linestyle="dashdot",
    label=f"Baseline Control $\\mathbf{{v_{{T,{LAYER}}}}}$"
)
# $\\mathbf{v_{T,cmd}}$
for a in ax: a.spines[['right', 'top']].set_visible(False)

ax[0].set_ylabel("$\\Delta H$ (mm)")
ax[1].set_ylabel("$v_T$ (mm/s)")

ax[1].set_xlabel("Segment Index")
ax[0].set_ylim(ylim)
ax[1].set_ylim([3-0.2,17+0.8])

# fig.suptitle("Tracking Performance Log-Log", fontsize=20)
# fig.tight_layout()
# plt.savefig(f"output_plots/{save_name_ll}", dpi=300)
# plt.show()

# fig, ax = plt.subplots(2,1, sharex=True)
ax[2].plot(
    dh_2,
    color=colors[1],
    label=f"LSTM Control $\\mathbf{{\\Delta H_{{{LAYER}}}}}$"
)
ax[2].plot(
    dh_d_2,
    color=colors[1],
    linestyle='dotted',
    label=f"LSTM Control $\\mathbf{{\\Delta H_{{d,{LAYER}}}}}$"
)
ax[3].plot(
    u_cmd_2,
    color=colors[1],
    linestyle="dashdot",
    label=f"LSTM Control $\\mathbf{{v_{{T,{LAYER}}}}}$"
)
# $\\mathbf{v_{T,cmd}}$
for a in ax: a.spines[['right', 'top']].set_visible(False)


ax[2].set_ylabel("$\\Delta H$ (mm)")
ax[3].set_ylabel("$v_T$ (mm/s)")

ax[3].set_xlabel("Segment Index")
ax[2].set_ylim(ylim)
ax[3].set_ylim([3-0.2,17+0.8])

lines_labels = [a.get_legend_handles_labels() for a in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
ax[0].legend(
    lines,
    labels,
    ncol=2,
    bbox_to_anchor=(0.5,1.00),
    loc='lower center'
)
# fig.text(-0.04,0.25, '(a)', va='center', rotation='vertical')
# fig.suptitle("Tracking Performance LSTM MPC", fontsize=20)
fig.tight_layout()
plt.savefig(f"output_plots/tracking_comb.png", dpi=300)
# plt.savefig(f"output_plots/{save_name_lstm}", dpi=300)
plt.show()
