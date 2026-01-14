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

dataset_b = torch.load("data/2026_01_12_11_48_35_wall_lstm_baseline_control.pt")
dataset_l = torch.load("data/2026_01_12_10_21_38_wall_lstm_control_a.pt")

# bulk prediction error metrics
errors_lc = []
errors_lid = []
errors_b = []

idxs_l = list(dataset_l.keys())[10:]
print(idxs_l)
idxs_b = list(dataset_b.keys())[10:]
print(idxs_b)

for idx in idxs_l:
    errors_lc.append(rms(dataset_l[idx]['truth'].numpy()-dataset_l[idx]['plant'].numpy()))
    errors_lid.append(rms(dataset_l[idx]['truth'].numpy()-dataset_l[idx]['cont'].numpy()))

for idx in idxs_b:
    errors_b.append(rms(dataset_l[idx]['truth'].numpy()-dataset_l[idx]['ll'].numpy()))

print(f"Conventional: {np.mean(errors_lc)}")
print(f"Innovation: {np.mean(errors_lid)}")
print(f"Log-Log: {np.mean(errors_b)}")
# plt.plot(errors_lc)
# plt.plot(errors_lid)
# plt.plot(errors_b)
# plt.show()
#SAMPLE PLOTS

test_idxs = [19, 79]

fig,ax = plt.subplots(4,1, sharex=True)
#lstm
for i, idx in enumerate(test_idxs):
    ax[i].plot(
        dataset_l[idx]['plant'],
        color=colors[2],
        label= "Conventional LSTM" if i==0 else None
    )
    ax[i].plot(
        dataset_l[idx]['cont'],
        color=colors[3],
        label= "Innovation-Driven LSTM" if i==0 else None
    )

    ax[i].plot(
        dataset_l[idx]['truth'],
        color=colors[0],
        linestyle='dotted',
    )
# baseline
for i, idx in enumerate(test_idxs):
    ax[i+2].plot(
        dataset_b[idx]['ll'],
        color=colors[1],
        label= "Log-Log Model" if i==0 else None
    )

    ax[i+2].plot(
        dataset_b[idx]['truth'],
        color=colors[0],
        linestyle='dotted',
        label= "Ground Truth" if i==0 else None
    )

ax[-1].set_xlabel("Segment Index")

for i, a in enumerate(ax):
    a.spines[['right', 'top']].set_visible(False)
    a.set_ylabel("$\\Delta H$ (mm)")
    a.set_ylim([0,3.1])

lines_labels = [a.get_legend_handles_labels() for a in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

labels = [labels[2], labels[0], labels[1], labels[3]]
lines = [lines[2], lines[0], lines[1], lines[3]]

ax[0].legend(
    lines,
    labels,
    ncol=2,
    bbox_to_anchor=(0.5,1.02),
    loc="lower center"
)

# add labels
ax[0].annotate(
    'Layer 20',
    xy=(0.01, 0.95),
    xycoords='axes fraction',
    ha='left',
    va='top'
)
ax[1].annotate(
    'Layer 80',
    xy=(0.01, 0.95),
    xycoords='axes fraction',
    ha='left',
    va='top'
)
ax[2].annotate(
    'Layer 20',
    xy=(0.01, 0.95),
    xycoords='axes fraction',
    ha='left',
    va='top'
)
ax[3].annotate(
    'Layer 80',
    xy=(0.01, 0.95),
    xycoords='axes fraction',
    ha='left',
    va='top'
)
fig.tight_layout()
plt.savefig(f"output_plots/model_verification_exp.png", dpi=300)
plt.savefig(f"output_plots/model_verification_exp.tiff", dpi=300)
# plt.savefig(f"output_plots/model_verification.tiff", dpi=300)
plt.show()
