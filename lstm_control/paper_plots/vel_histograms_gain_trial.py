import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

# colors = ["#7fc97f","#beaed4","#fdc086"]
colors = ["#66c2a5","#fc8d62","#8da0cb"]

DATA_DIR = "data/"
# load baseline test set
def rms(x):
    return np.sqrt(np.sum(np.square(x))/len(x))


A_IDX = 8
B_IDX = 8
DH_IDX = 4
##########
test_data_1 = torch.load(f"{DATA_DIR}/20260107-043810_gain_tests/test_results.pt")
vel_data = test_data_1["velocity"] 
####################
fig, ax = plt.subplots(1,1)
bins = ax.hist(
    vel_data[B_IDX, A_IDX, DH_IDX, :,:,:].reshape(-1),
    bins=20,
    density=True,
    color=colors[0],
    rwidth=0.95
)[1]
# labels
# ax[0,0].set_ylim(ax[1,0].set_ylim(ax[2,0].set_ylim(ax[0,1].set_ylim(ax[1,1].set_ylim(ax[2,1].set_ylim())))))
# ax[2].set_xlabel("Command Velocity (mm/s)")
ax.spines[['right', 'top']].set_visible(False)
ax.set_ylabel("Density")
ax.set_xlabel("Command Velocity (mm/s)")

fig.suptitle("Commanded Velocity Distributions", fontsize=20)
fig.tight_layout()
plt.show()
# fig, ax = plt.subplots(3,2)
# bins = ax[0,0].hist(
#     test_data_1["u_cmd_all"].reshape(-1),
#     bins=20,
#     density=True,
#     color=colors[0],
#     rwidth=0.95
# )[1]
# ax[1,0].hist(
#     test_data_2["u_cmd_all"].reshape(-1),
#     bins=bins,
#     density=True,
#     color=colors[1],
#     rwidth=0.95
# )
# ax[2,0].hist(
#     test_data_3["u_cmd_all"].reshape(-1),
#     bins=bins,
#     density=True,
#     color=colors[2],
#     rwidth=0.95
# )
# bins = ax[0,1].hist(
#     test_data_1n["u_cmd_all"].reshape(-1),
#     bins=20,
#     density=True,
#     color=colors[0],
#     rwidth=0.95
# )[1]
# ax[1,1].hist(
#     test_data_2n["u_cmd_all"].reshape(-1),
#     bins=bins,
#     density=True,
#     color=colors[1],
#     rwidth=0.95
# )
# ax[2,1].hist(
#     test_data_3n["u_cmd_all"].reshape(-1),
#     bins=bins,
#     density=True,
#     color=colors[2],
#     rwidth=0.95
# )

# # labels
# # ax[0,0].set_ylim(ax[1,0].set_ylim(ax[2,0].set_ylim(ax[0,1].set_ylim(ax[1,1].set_ylim(ax[2,1].set_ylim())))))
# # ax[2].set_xlabel("Command Velocity (mm/s)")
# ax[0,0].set_title("Log-Log Baseline")
# ax[1,0].set_title("Single-Layer LSTM MPC")
# ax[2,0].set_title("Multi-Layer LSTM MPC")
# ax[0,1].set_title("Log-Log Baseline - Noise")
# ax[1,1].set_title("Single-Layer LSTM MPC - Noise")
# ax[2,1].set_title("Multi-Layer LSTM MPC - Noise")

# for row_a in ax:
#     for a in row_a:
#         a.spines[['right', 'top']].set_visible(False)
#         a.set_ylabel("Density")
#         a.set_xlabel("Command Velocity (mm/s)")

# fig.suptitle("Commanded Velocity Distributions", fontsize=20)
# fig.tight_layout()
# plt.savefig(f"output_plots/{save_name}", dpi=300)
# plt.show()
