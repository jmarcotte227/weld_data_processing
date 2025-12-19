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

# baseline
test_data_1 = torch.load(f"{DATA_DIR}Log-Log_Baseline_plantfb_False_pmodel_h-8_part-0_loss-0.4866_cmodel_h-8_part-1_loss-0.041120251218-233856/test_results.pt")

# Single Layer LSTM
test_data_2 = torch.load(f"{DATA_DIR}Linearized_QP_Control_plantfb_False_pmodel_h-8_part-0_loss-0.4866_cmodel_h-8_part-1_loss-0.041120251218-233750/test_results.pt")

# multi-layer LSTM
test_data_3 = torch.load(f"{DATA_DIR}/Linearized_QP_Control_plantfb_False_pmodel_h-8_part-0_loss-0.4866_cmodel_h-8_part-1_loss-0.041120251218-235015/test_results.pt")

fig, ax = plt.subplots(3,1)
bins = ax[0].hist(
    test_data_1["u_cmd_all"].reshape(-1),
    bins=20,
    density=True,
    color=colors[0],
    rwidth=0.95
)[1]
ax[1].hist(
    test_data_2["u_cmd_all"].reshape(-1),
    bins=bins,
    density=True,
    color=colors[1],
    rwidth=0.95
)
ax[2].hist(
    test_data_3["u_cmd_all"].reshape(-1),
    bins=bins,
    density=True,
    color=colors[2],
    rwidth=0.95
)

# labels
ax[0].set_ylabel("Density")
ax[0].set_ylim(ax[1].set_ylim())
ax[2].set_xlabel("Command Velocity (mm/s)")
ax[0].set_title("Log-Log Baseline")
ax[1].set_title("Single-Layer LSTM MPC")
ax[2].set_title("Multi-Layer LSTM MPC")

for a in ax:
    a.spines[['right', 'top']].set_visible(False)
    a.set_ylabel("Density")
    a.set_xlabel("Command Velocity (mm/s)")

fig.suptitle("Commanded Velocity Distributions", fontsize=20)
fig.tight_layout()
plt.savefig("output_plots/vel_histograms.png", dpi=300)
plt.show()
