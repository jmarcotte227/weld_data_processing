import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns

def rms(x):
    return np.sqrt(np.sum(np.square(x))/len(x))

rc('text', usetex=True)
rc('font',**{'family':'sans-serif','sans-serif':['Latin Modern Sans']})

# colors = ["#7fc97f","#beaed4","#fdc086"]
# colors = ["#66c2a5","#fc8d62","#8da0cb"]
colors = [
    '#e69f00',
    '#009e73',
]

DATA_DIR = "../../../recorded_data/"
PROC_DATA_DIR = "../data/"
# load baseline test set

##### WITH NOISE #####
# baseline
DATA_1 = "2026_01_12_11_48_35_wall_lstm_baseline_control"
DATA_2 = "2026_01_12_10_21_38_wall_lstm_control"

NUM_LAYERS = 100
####################

fig, ax = plt.subplots(1,1)

# baseline
errors_1 = np.zeros(NUM_LAYERS-1)

dh = np.loadtxt(f"{PROC_DATA_DIR}calc_dh/{DATA_1}_dh.csv", delimiter=',') 
# calculate rms at each layer
for layer in range(1,NUM_LAYERS):
    dh_d = np.loadtxt(f"{DATA_DIR}{DATA_1}/layer_{layer}/dh_d.csv", delimiter=',')
    layer_error = dh_d-dh[layer]
    if layer ==1: print(layer_error)
    # errors_1[layer] = rms(layer_error)
    errors_1[layer-1] = rms(layer_error[~np.isnan(layer_error)])

ax.scatter(
    np.linspace(1,99,99),
    errors_1,
    color=colors[0],
    # marker=markers[idx],
)

# LSTM control
errors_2 = np.zeros(NUM_LAYERS-1)

dh = np.loadtxt(f"{PROC_DATA_DIR}calc_dh/{DATA_2}_dh.csv", delimiter=',') 
# calculate rms at each layer
for layer in range(1,NUM_LAYERS):
    dh_d = np.loadtxt(f"{DATA_DIR}{DATA_2}/layer_{layer}/dh_d.csv", delimiter=',')
    layer_error = dh_d-dh[layer]
    if layer ==1: print(layer_error)
    errors_2[layer-1] = rms(layer_error[~np.isnan(layer_error)])

print(len(errors_2[9:]))
print(f"Baseline AVG RMSE 90: {np.mean(errors_1[9:])}")
print(f"LSTM AVG RMSE 90: {np.mean(errors_2[9:])}")

ax.scatter(
    np.linspace(1,99,99),
    errors_2,
    color=colors[1],
    # marker=markers[idx],
)

# labels
ax.set_ylabel("RMSE (mm)")
ax.set_xlabel("Layer Number")
ax.spines[['right', 'top']].set_visible(False)
# ax.legend(["Log-Log Baseline","Log-Log Baseline Noise", "LSTM MPC", "LSTM MPC Noise"])
ax.legend(
    [
        "Baseline Control",
        "LSTM Control",
    ],
    ncol=2,
    bbox_to_anchor=(0.5,1.02),
    loc="lower center"
)
ax.set_ylim([0,1.6])

# fig.suptitle("Reference Tracking Error", fontsize=20)
fig.set_size_inches(6.4, 3)
fig.tight_layout()
plt.savefig("output_plots/exp_tracking.png", dpi=300)
plt.savefig("output_plots/exp_tracking.tiff", dpi=300)
plt.show()

fig,ax=plt.subplots()
ax.hist(
    errors_all.reshape(-1),
    bins=100,
    density=True,
)
ax.set_xlim([-3.25,3.25])
plt.show()

