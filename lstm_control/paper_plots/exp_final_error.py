import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns

rc('text', usetex=True)
rc('font',**{'family':'sans-serif','sans-serif':['Latin Modern Sans']})

# colors = ["#7fc97f","#beaed4","#fdc086"]
# colors = ["#66c2a5","#fc8d62","#8da0cb"]
colors = [
    '#e69f00',
    '#009e73',
]

def rms(x):
    return np.sqrt(np.sum(np.square(x))/len(x))
DATA_DIR = "../../../recorded_data/"
PROC_DATA_DIR = "../data/"
# load baseline test set

##### WITH NOISE #####
# baseline
DATA_1 = "2026_01_12_11_48_35_wall_lstm_baseline_control"
DATA_2 = "2026_01_12_10_21_38_wall_lstm_control"

NUM_LAYERS = 100

# fig, ax = plt.subplots(2,1, sharex=True)
# for a in ax:
#     a.plot(
#         test_data_1["H"][-1]-test_data_1["H_d"][-1],
#         color=colors[0],
#     )
#     a.plot(
#         test_data_2["H"][-1]-test_data_2["H_d"][-1],
#         color=colors[1],
#     )
#     a.plot(
#         test_data_3["H"][-1]-test_data_3["H_d"][-1],
#         color=colors[2],
#     )
#     a.plot(
#         test_data_1n["H"][-1]-test_data_1n["H_d"][-1],
#         color=colors[0],
#         linestyle='dashed'
#     )
#     a.plot(
#         test_data_2n["H"][-1]-test_data_2n["H_d"][-1],
#         color=colors[1],
#         linestyle='dashed'
#     )
#     a.plot(
#         test_data_3n["H"][-1]-test_data_3n["H_d"][-1],
#         color=colors[2],
#         linestyle='dashed'
#     )

#     a.spines[['right', 'top']].set_visible(False)

H_1 = np.loadtxt(f"{PROC_DATA_DIR}/calc_h/{DATA_1}_h.csv", delimiter=',')
H_2 = np.loadtxt(f"{PROC_DATA_DIR}/calc_h/{DATA_2}_h.csv", delimiter=',')

fig, ax = plt.subplots(1,1, sharex=True)
ax.plot(
    H_1[-1]-163,
    color=colors[0],
)
ax.plot(
    H_2[-1]-163,
    color=colors[1],
)

ax.spines[['right', 'top']].set_visible(False)

# labels
# ax[0].set_ylabel("Height Error (mm)")
# ax[1].set_ylabel("Height Error (mm)")
# ax[1].set_xlabel("Segment Index")
# ax[0].legend(["Log-Log Baseline","Single Layer LSTM MPC", "Multi-Layer LSTM MPC",
#            "Log-Log Baseline Noise","Single Layer LSTM MPC Noise", "Multi-Layer LSTM MPC Noise"],
#              ncol=2)

# ax[1].set_ylim([-1,1])
# fig.suptitle("Final Layer Height Error", fontsize=20)
# fig.tight_layout()
# plt.savefig(f"output_plots/{save_name}", dpi=300)
# plt.show()
ax.set_ylabel("Height Error (mm)")
ax.set_xlabel("Segment Index")
ax.grid()
# ax.set_ylim([-1,4])
ax.legend(
    ["Baseline Control","LSTM Control"],
    ncol=2,
    bbox_to_anchor=(0.5,1.02),
    loc='lower center'
)

# fig.suptitle("Final Layer Height Error", fontsize=20)
fig.set_size_inches(6.4, 3)
fig.tight_layout()
plt.savefig(f"output_plots/exp_final_err.png", dpi=300)
plt.savefig(f"output_plots/exp_final_err.tiff", dpi=300)
plt.show()

# average RMSE stats
H_d = np.linspace(9.55, 163, NUM_LAYERS)
print(H_d)
errors_1 = []
errors_2 = []
for layer in range(10,NUM_LAYERS):
    errors_1.append(rms(H_1[layer][~np.isnan(H_1[layer])]-H_d[layer]))
    errors_2.append(rms(H_2[layer][~np.isnan(H_2[layer])]-H_d[layer]))
# errors
print("---Final---")
print(f"Test 1: {rms(H_1[-1]-163)}")
print(f"Test 2: {rms(H_2[-1]-163)}")
print(f"Percent Change: {100*(rms(H_2[-1]-163)-rms(H_1[-1]-163))/rms(H_1[-1]-163)}")

print("---Over Last 90 layers---")
print(len(errors_1))
print(f"Test 1: {np.mean(errors_1)}")
print(f"Test 2: {np.mean(errors_2)}")
