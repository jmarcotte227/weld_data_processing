import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns


rc('text', usetex=True)
rc('font',**{'family':'sans-serif','sans-serif':['Latin Modern Sans']})
# colors = ["#7fc97f","#beaed4","#fdc086"]
# colors = [
    # '#e69f00',
    # '#009e73',
# ]
colors = [
    '#0072b2',
    '#e69f00',
    '#d55e00',
    '#009e73',
]

dataset = torch.load("data/041015model_h-8_part-0_loss-0.2000model_h-8_part-1_loss-0.0684.pt")

idxs = dataset.keys()

fig,ax = plt.subplots(len(idxs),1, sharex=True)
for i in idxs:
    labels = []
    for j, cont in enumerate(['ll', 'plant', 'cont']):
        print(cont)
        ax[i].plot(dataset[i][cont], color=colors[j+1])
        labels.append(cont)

    ax[i].plot(dataset[i]['truth'], color=colors[0], linestyle='dotted')

ax[-1].set_xlabel("Segment Index")
for i, a in enumerate(ax):
    a.spines[['right', 'top']].set_visible(False)
    a.set_ylabel("$\\Delta H$ (mm)")
    a.set_ylim([0,3.1])
ax[0].legend(
    [
        "Log-Log Model",
        "Conventional LSTM",
        "Innovation Driven LSTM",
        "Ground Truth",
    ],
    ncol=2,
    bbox_to_anchor=(0.5,1.02),
    loc="lower center"
)
fig.tight_layout()
plt.savefig(f"output_plots/model_verification.png", dpi=300)
plt.savefig(f"output_plots/model_verification.tiff", dpi=300)
plt.show()
