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

dataset = torch.load("data/041015model_h-8_part-0_loss-0.2000model_h-8_part-1_loss-0.0684.pt")

idxs = dataset.keys()

fig,ax = plt.subplots(len(idxs),1, sharex=True)
for i in idxs:
    labels = []
    for cont in dataset[i].keys():
        ax[i].plot(dataset[i][cont])
        labels.append(cont)

ax[-1].set_xlabel("Segment Index")
for a in ax:
    a.spines[['right', 'top']].set_visible(False)
    a.set_ylabel("$\\Delta H$ (mm)")
    a.set_ylim([0,3.1])
ax[0].legend(
    [
        "Ground Truth",
        "Conventional LSTM",
        "Innovation Driven LSTM",
        "Log-Log Model"
    ],
    ncol=2,
    bbox_to_anchor=(0.5,1.02),
    loc="lower center"
)
fig.tight_layout()
plt.savefig(f"output_plots/model_verification.png", dpi=300)
plt.show()
