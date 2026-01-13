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

DATA_DIR = "data/"
# load baseline test set

##### NO NOISE #####
# baseline
test_data_1 = torch.load(f"{DATA_DIR}Log-Log_Baseline_noise_False_pmodel_h-8_part-0_loss-0.2000_cmodel_h-8_part-1_loss-0.068420260107-133050/test_results.pt")

# Single Layer LSTM
test_data_2 = torch.load(f"{DATA_DIR}Linearized_QP_Control_noise_False_pmodel_h-8_part-0_loss-0.2000_cmodel_h-8_part-1_loss-0.068420260107-132840/test_results.pt")

save_name = 'final_error.png'
####################
##### WITH NOISE #####
# baseline
test_data_1n = torch.load(f"{DATA_DIR}Log-Log_Baseline_noise_True_pmodel_h-8_part-0_loss-0.2000_cmodel_h-8_part-1_loss-0.068420260107-132945/test_results.pt")

# Single Layer LSTM
test_data_2n = torch.load(f"{DATA_DIR}Linearized_QP_Control_noise_True_pmodel_h-8_part-0_loss-0.2000_cmodel_h-8_part-1_loss-0.068420260107-132812/test_results.pt")
####################

datasets = [test_data_1, test_data_1n, test_data_2, test_data_2n]
# color_choice = [colors[0], colors[0], colors[1], colors[1]]
# markers=['o', 'o', 'o', 'o']
face_colors = ['none', colors[0], 'none', colors[1]]
edge_colors = [colors[0], colors[0], colors[1], colors[1]]

fig, ax = plt.subplots(1,1)
for idx, d_set in enumerate(datasets):
    errors = np.zeros(d_set["num_layers"])

    # calculate rms at each layer
    for layer in range(d_set["num_layers"]):
        layer_error = d_set["dh_d"][layer].numpy()-d_set["dh"][layer].numpy()
        errors[layer] = rms(layer_error)

    ax.scatter(
        np.linspace(1,100,100),
        errors,
        # color=color_choice[idx],
        # marker=markers[idx],
        edgecolors=edge_colors[idx],
        facecolors=face_colors[idx],
    )
    print("Tracking error 90: ", np.mean(errors[10:]))

# labels
ax.set_ylabel("RMSE (mm)")
ax.set_xlabel("Layer Number")
ax.spines[['right', 'top']].set_visible(False)
# ax.legend(["Log-Log Baseline","Log-Log Baseline Noise", "LSTM MPC", "LSTM MPC Noise"])
ax.legend(
    [
        "Baseline Control Without Noise",
        "Baseline Control With Noise",
        "LSTM Control Without Noise",
        "LSTM Control With Noise"
    ],
    ncol=2,
    bbox_to_anchor=(0.5,1.02),
    loc="lower center"
)

# fig.suptitle("Reference Tracking Error", fontsize=20)
fig.set_size_inches(6.4, 3)
fig.tight_layout()
plt.savefig("output_plots/tracking.png", dpi=300)
plt.savefig("output_plots/tracking.tiff", dpi=300)
plt.show()

fig,ax=plt.subplots()
ax.hist(
    errors_all.reshape(-1),
    bins=100,
    density=True,
)
ax.set_xlim([-3.25,3.25])
plt.show()

