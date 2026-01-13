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

DATA_DIR = "data/"
# load baseline test set

##### NO NOISE #####
# baseline
test_data_1 = torch.load(f"{DATA_DIR}Log-Log_Baseline_noise_False_pmodel_h-8_part-0_loss-0.2000_cmodel_h-8_part-1_loss-0.068420260107-133050/test_results.pt")

# Single Layer LSTM
test_data_2 = torch.load(f"{DATA_DIR}Linearized_QP_Control_noise_False_pmodel_h-8_part-0_loss-0.2000_cmodel_h-8_part-1_loss-0.068420260107-132840/test_results.pt")

save_name = 'final_error'
####################
##### WITH NOISE #####
# baseline
test_data_1n = torch.load(f"{DATA_DIR}Log-Log_Baseline_noise_True_pmodel_h-8_part-0_loss-0.2000_cmodel_h-8_part-1_loss-0.068420260107-132945/test_results.pt")

# Single Layer LSTM
test_data_2n = torch.load(f"{DATA_DIR}Linearized_QP_Control_noise_True_pmodel_h-8_part-0_loss-0.2000_cmodel_h-8_part-1_loss-0.068420260107-132812/test_results.pt")
####################

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
fig, ax = plt.subplots(1,1, sharex=True)
ax.plot(
    test_data_1["H"][-1]-test_data_1["H_d"][-1],
    color=colors[0],
    linestyle='dashed'
)
ax.plot(
    test_data_2["H"][-1]-test_data_2["H_d"][-1],
    color=colors[1],
    linestyle='dashed'
)
ax.plot(
    test_data_1n["H"][-1]-test_data_1n["H_d"][-1],
    color=colors[0],
)
ax.plot(
    test_data_2n["H"][-1]-test_data_2n["H_d"][-1],
    color=colors[1],
)

ax.spines[['right', 'top']].set_visible(False)

# errors
print("---Full---")
print(f"Test 1: {rms((test_data_1['H'][-1]-test_data_1['H_d'][-1]).numpy())}")
print(f"Test 2: {rms((test_data_2['H'][-1]-test_data_2['H_d'][-1]).numpy())}")
print("---Trimmed---")
print(f"Test 1: {rms((test_data_1['H'][-1][1:-1]-test_data_1['H_d'][-1][1:-1]).numpy())}")
print(f"Test 2: {rms((test_data_2['H'][-1][1:-1]-test_data_2['H_d'][-1][1:-1]).numpy())}")

print("Noise")
print("Severly Trimmed")
print(len(test_data_1n['H'][-1][6:-6]))
print(f"Test 1 N: {rms((test_data_1n['H'][-1][6:-6]-test_data_1n['H_d'][-1][6:-6]).numpy())}")
print(f"Test 1 N: {rms((test_data_1['H'][-1][6:-6]-test_data_1['H_d'][-1][6:-6]).numpy())}")

print("---Full---")
print(f"Test 1 N: {rms((test_data_1n['H'][-1]-test_data_1n['H_d'][-1]).numpy())}")
print(f"Test 2 N: {rms((test_data_2n['H'][-1]-test_data_2n['H_d'][-1]).numpy())}")
print("---Trimmed---")
print(f"Test 1 N: {rms((test_data_1n['H'][-1][1:-1]-test_data_1n['H_d'][-1][1:-1]).numpy())}")
print(f"Test 2 N: {rms((test_data_2n['H'][-1][1:-1]-test_data_2n['H_d'][-1][1:-1]).numpy())}")

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
    ["Baseline Control Without Noise","LSTM Control Without Noise",
     "Baseline Control With Noise","LSTM Control With Noise"],
    ncol=2,
    bbox_to_anchor=(0.5,1.02),
    loc='lower center'
)

# fig.suptitle("Final Layer Height Error", fontsize=20)
fig.set_size_inches(6.4, 3)
fig.tight_layout()
plt.savefig(f"output_plots/{save_name}.png", dpi=300)
plt.savefig(f"output_plots/{save_name}.tiff", dpi=300)
plt.show()
