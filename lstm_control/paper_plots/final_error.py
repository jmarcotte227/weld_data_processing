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
# load baseline test set

##### NO NOISE #####
# baseline
test_data_1 = torch.load(f"{DATA_DIR}Log-Log_Baseline_plantfb_False_pmodel_h-8_part-0_loss-0.4866_cmodel_h-8_part-1_loss-0.041120251218-233856/test_results.pt")

# Single Layer LSTM
test_data_2 = torch.load(f"{DATA_DIR}Linearized_QP_Control_plantfb_False_pmodel_h-8_part-0_loss-0.4866_cmodel_h-8_part-1_loss-0.041120251218-233750/test_results.pt")

# multi-layer LSTM
test_data_3 = torch.load(f"{DATA_DIR}/Linearized_QP_Control_plantfb_False_pmodel_h-8_part-0_loss-0.4866_cmodel_h-8_part-1_loss-0.041120251218-235015/test_results.pt")
save_name = 'final_error.png'
####################
##### WITH NOISE #####
# baseline
test_data_1n = torch.load(f"{DATA_DIR}Log-Log_Baseline_plantfb_False_pmodel_h-8_part-0_loss-0.4866_cmodel_h-8_part-1_loss-0.041120251219-100212/test_results.pt")

# Single Layer LSTM
test_data_2n = torch.load(f"{DATA_DIR}Linearized_QP_Control_plantfb_False_pmodel_h-8_part-0_loss-0.4866_cmodel_h-8_part-1_loss-0.041120251219-100304/test_results.pt")

# multi-layer LSTM
test_data_3n = torch.load(f"{DATA_DIR}Linearized_QP_Control_plantfb_False_pmodel_h-8_part-0_loss-0.4866_cmodel_h-8_part-1_loss-0.041120251219-100616/test_results.pt")
# save_name = 'final_error_noise.png'
####################

fig, ax = plt.subplots(2,1, sharex=True)
for a in ax:
    a.plot(
        test_data_1["H"][-1]-test_data_1["H_d"][-1],
        color=colors[0],
    )
    a.plot(
        test_data_2["H"][-1]-test_data_2["H_d"][-1],
        color=colors[1],
    )
    a.plot(
        test_data_3["H"][-1]-test_data_3["H_d"][-1],
        color=colors[2],
    )
    a.plot(
        test_data_1n["H"][-1]-test_data_1n["H_d"][-1],
        color=colors[0],
        linestyle='dashed'
    )
    a.plot(
        test_data_2n["H"][-1]-test_data_2n["H_d"][-1],
        color=colors[1],
        linestyle='dashed'
    )
    a.plot(
        test_data_3n["H"][-1]-test_data_3n["H_d"][-1],
        color=colors[2],
        linestyle='dashed'
    )

    a.spines[['right', 'top']].set_visible(False)

# errors
print("---Full---")
print(f"Test 1: {rms((test_data_1['H'][-1]-test_data_1['H_d'][-1]).numpy())}")
print(f"Test 2: {rms((test_data_2['H'][-1]-test_data_2['H_d'][-1]).numpy())}")
print(f"Test 3: {rms((test_data_3['H'][-1]-test_data_3['H_d'][-1]).numpy())}")
print("---Trimmed---")
print(f"Test 1: {rms((test_data_1['H'][-1][1:-1]-test_data_1['H_d'][-1][1:-1]).numpy())}")
print(f"Test 2: {rms((test_data_2['H'][-1][1:-1]-test_data_2['H_d'][-1][1:-1]).numpy())}")
print(f"Test 3: {rms((test_data_3['H'][-1][1:-1]-test_data_3['H_d'][-1][1:-1]).numpy())}")

print()
print(f"Num layers: {test_data_3['H'].shape}")


# labels
ax[0].set_ylabel("Height Error (mm)")
ax[1].set_ylabel("Height Error (mm)")
ax[1].set_xlabel("Segment Index")
ax[0].legend(["Log-Log Baseline","Single Layer LSTM MPC", "Multi-Layer LSTM MPC",
           "Log-Log Baseline Noise","Single Layer LSTM MPC Noise", "Multi-Layer LSTM MPC Noise"])

ax[1].set_ylim([-2.5,2.5])
fig.suptitle("Final Layer Height Error", fontsize=20)
fig.tight_layout()
plt.savefig(f"output_plots/{save_name}", dpi=300)
plt.show()
