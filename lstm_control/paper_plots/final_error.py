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

fig, ax = plt.subplots(1,1)
ax.plot(
    test_data_1["H"][-1]-test_data_1["H_d"][-1],
    color=colors[0],
)
ax.plot(
    test_data_2["H"][-1]-test_data_2["H_d"][-1],
    color=colors[1],
)
ax.plot(
    test_data_3["H"][-1]-test_data_2["H_d"][-1],
    color=colors[2],
)

# labels
ax.set_ylabel("Height Error (mm)")
ax.set_xlabel("Segment Index")
ax.spines[['right', 'top']].set_visible(False)
ax.legend(["Log-Log Baseline","Single Layer LSTM MPC", "Multi-Layer LSTM MPC"])

fig.suptitle("Final Layer Height Error", fontsize=20)
fig.tight_layout()
plt.savefig("output_plots/final_error.png", dpi=300)
plt.show()
