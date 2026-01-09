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

def rms(x):
    return np.sqrt(np.sum(np.square(x))/len(x))

DATA_DIR = "data/"
LAYER = 46
# load baseline test set

##### NO NOISE #####
# baseline
test_data_1 = torch.load(f"{DATA_DIR}Log-Log_Baseline_noise_False_pmodel_h-8_part-0_loss-0.2000_cmodel_h-8_part-1_loss-0.068420260107-133050/test_results.pt")

# Single Layer LSTM
test_data_2 = torch.load(f"{DATA_DIR}Linearized_QP_Control_noise_False_pmodel_h-8_part-0_loss-0.2000_cmodel_h-8_part-1_loss-0.068420260107-132840/test_results.pt")

save_name_ll = 'ref_tracking_ll.png'
save_name_lstm = 'ref_tracking_lstm.png'
####################
##### WITH NOISE #####
# baseline
test_data_1n = torch.load(f"{DATA_DIR}Log-Log_Baseline_noise_True_pmodel_h-8_part-0_loss-0.2000_cmodel_h-8_part-1_loss-0.068420260107-132945/test_results.pt")

# Single Layer LSTM
test_data_2n = torch.load(f"{DATA_DIR}Linearized_QP_Control_noise_True_pmodel_h-8_part-0_loss-0.2000_cmodel_h-8_part-1_loss-0.068420260107-132812/test_results.pt")
####################

ylim = [
    min([
        torch.min(test_data_1n["dh"][LAYER]),
        torch.min(test_data_1n["dh_d"][LAYER]),
        torch.min(test_data_2n["dh"][LAYER]),
        torch.min(test_data_2n["dh_d"][LAYER]),
    ])-0.1,
    max([
        torch.max(test_data_1n["dh"][LAYER]),
        torch.max(test_data_1n["dh_d"][LAYER]),
        torch.max(test_data_2n["dh"][LAYER]),
        torch.max(test_data_2n["dh_d"][LAYER]),
    ])+0.1
]

# Log Log Plots
fig, ax = plt.subplots(2,1, sharex=True)
ax[0].plot(
    test_data_1n["dh"][LAYER],
    color=colors[0],
)
ax[0].plot(
    test_data_1n["dh_d"][LAYER],
    color=colors[0],
    linestyle='dotted'
)
ax[1].plot(
    test_data_1n["u_cmd_all"][LAYER],
    color=colors[0],
)
# $\\mathbf{v_{T,cmd}}$
for a in ax: a.spines[['right', 'top']].set_visible(False)

ax[0].set_ylabel("Deposition Height (mm)")
ax[1].set_ylabel("Commanded Velocity (mm/s)")

ax[1].set_xlabel("Segment Index")
ax[0].legend(
    [f"$\\mathbf{{\\Delta H_{{{LAYER}}}}}$", f"$\\mathbf{{\\Delta H_{{d,{LAYER}}}}}$"],
)
ax[1].legend(
        [f"$\\mathbf{{v_{{T,{LAYER}}}}}$"],
)
ax[0].set_ylim(ylim)
ax[1].set_ylim([3-0.1,17+0.1])

fig.suptitle("Tracking Performance Log-Log", fontsize=20)
fig.tight_layout()
plt.savefig(f"output_plots/{save_name_ll}", dpi=300)
plt.show()

fig, ax = plt.subplots(2,1, sharex=True)
ax[0].plot(
    test_data_2n["dh"][LAYER],
    color=colors[1],
)
ax[0].plot(
    test_data_2n["dh_d"][LAYER],
    color=colors[1],
    linestyle='dotted'
)
ax[1].plot(
    test_data_2n["u_cmd_all"][LAYER],
    color=colors[1],
)
# $\\mathbf{v_{T,cmd}}$
for a in ax: a.spines[['right', 'top']].set_visible(False)

ax[0].set_ylabel("Deposition Height (mm)")
ax[1].set_ylabel("Commanded Velocity (mm/s)")

ax[1].set_xlabel("Segment Index")
ax[0].legend(
    [f"$\\mathbf{{\\Delta H_{{{LAYER}}}}}$", f"$\\mathbf{{\\Delta H_{{d,{LAYER}}}}}$"],
)
ax[1].legend(
        [f"$\\mathbf{{v_{{T,{LAYER}}}}}$"],
)
ax[0].set_ylim(ylim)
ax[1].set_ylim([3-0.1,17+0.1])

fig.suptitle("Tracking Performance LSTM MPC", fontsize=20)
fig.tight_layout()
plt.savefig(f"output_plots/{save_name_lstm}", dpi=300)
plt.show()
