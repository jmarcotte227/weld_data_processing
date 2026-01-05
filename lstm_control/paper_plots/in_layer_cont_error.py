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

test_data= torch.load(f"{DATA_DIR}Linearized_QP_Control_noise_True_pmodel_h-8_part-0_loss-0.2000_cmodel_h-8_part-1_loss-0.068420260105-135439/test_results.pt")
# test_data= torch.load(f"{DATA_DIR}Linearized_QP_Control_noise_False_pmodel_h-8_part-0_loss-0.2000_cmodel_h-8_part-1_loss-0.068420260105-134846/test_results.pt")



errors = np.zeros(test_data["num_layers"])
# calculate rms at each layer
fig,ax = plt.subplots(1,1)
for layer in range(test_data["num_layers"]):
    layer_error = test_data["dh_d"][layer].numpy()-test_data["dh"][layer].numpy()
    ax.plot(layer_error)
    errors[layer] = rms(layer_error)

plt.show()
fig, ax = plt.subplots(1,1)
ax.scatter(
    np.linspace(1,50,50),
    errors,
)

# labels
ax.set_ylabel("Height Error (mm)")
ax.set_xlabel("Segment Index")
ax.spines[['right', 'top']].set_visible(False)
ax.legend(["Log-Log Baseline","Single Layer LSTM MPC"])

fig.suptitle("Final Layer Height Error", fontsize=20)
fig.tight_layout()
plt.savefig("output_plots/final_error.png", dpi=300)
plt.show()
