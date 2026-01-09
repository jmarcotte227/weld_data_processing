import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns

def rms(x):
    return np.sqrt(np.sum(np.square(x))/len(x))

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

# colors = ["#7fc97f","#beaed4","#fdc086"]
colors = ["#66c2a5","#fc8d62","#8da0cb"]

DATA_DIR = "data/"
# load baseline test set

test_data_n= torch.load(f"{DATA_DIR}Linearized_QP_Control_noise_True_pmodel_h-8_part-0_loss-0.2000_cmodel_h-8_part-1_loss-0.068420260105-135439/test_results.pt")
test_data= torch.load(f"{DATA_DIR}Linearized_QP_Control_noise_False_pmodel_h-8_part-0_loss-0.2000_cmodel_h-8_part-1_loss-0.068420260105-134846/test_results.pt")


fig, ax = plt.subplots(1,1)
errors = np.zeros(test_data["num_layers"])
errors_all = np.zeros_like(test_data["dh_d"])
# calculate rms at each layer
# fig,ax = plt.subplots(1,1)
for layer in range(test_data["num_layers"]):
    layer_error = test_data["dh_d"][layer].numpy()-test_data["dh"][layer].numpy()
    # ax.plot(layer_error)
    errors[layer] = rms(layer_error)

# ax.set_xlabel("Seg Idx")
# ax.set_ylabel("dH Error (mm)")
# plt.show()
ax.scatter(
    np.linspace(1,50,50),
    errors,
)
# fig_n, ax_n = plt.subplots(1,1)
for layer in range(test_data_n["num_layers"]):
    layer_error = test_data_n["dh_d"][layer].numpy()-test_data_n["dh"][layer].numpy()
    # ax_n.plot(layer_error)
    errors[layer] = rms(layer_error)
    errors_all[layer,:] = layer_error

# ax_n.set_xlabel("Seg Idx")
# ax_n.set_ylabel("dH Error (mm)")
ax.scatter(
    np.linspace(1,50,50),
    errors,
)

# labels
ax.set_ylabel("dH Error (mm)")
ax.set_xlabel("Layer")
ax.spines[['right', 'top']].set_visible(False)
ax.legend(["No Noise","Noise Applied"])

fig.suptitle("dH Error", fontsize=20)
fig.tight_layout()
plt.savefig("output_plots/final_error.png", dpi=300)
plt.show()

fig,ax=plt.subplots()
ax.hist(
    errors_all.reshape(-1),
    bins=100,
    density=True,
)
ax.set_xlim([-3.25,3.25])
plt.show()

