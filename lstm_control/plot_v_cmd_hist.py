import numpy as np
import matplotlib.pyplot as plt

num_bins = 30

bins = np.linspace(3,17,num_bins)
baseline_v_set = np.loadtxt(
    "data/v_set/2026_02_25_09_57_16_tube_baseline_control_v_cmd.csv",
    delimiter=','
)
lstm_v_set = np.loadtxt(
    "data/v_set/2026_02_23_11_23_56_tube_lstm_control_v_cmd.csv",
    delimiter=','
)
fig,ax = plt.subplots(2,1)
hist_values = np.zeros((baseline_v_set.shape[0], num_bins-1))
for layer in range(baseline_v_set.shape[0]):
    hist_data, edges = np.histogram(baseline_v_set[layer], bins=bins)
    hist_values[layer,:] = hist_data
im=ax[0].imshow(
    hist_values.T,
    origin="lower",
    extent=[0,baseline_v_set.shape[0], bins[0], bins[-1]]
)
plt.colorbar(im)
ax[0].set_yticks(edges[::5])
hist_values = np.zeros((lstm_v_set.shape[0], num_bins-1))
for layer in range(lstm_v_set.shape[0]):
    hist_data, edges = np.histogram(lstm_v_set[layer], bins=bins)
    hist_values[layer,:] = hist_data
im=ax[1].imshow(
    hist_values.T,
    origin="lower",
    extent=[0,lstm_v_set.shape[0], bins[0], bins[-1]]
)
plt.colorbar(im)
ax[1].set_yticks(edges[::5])
ax[0].set_ylabel("Velocity Counts")
ax[1].set_ylabel("Velocity Counts")
ax[0].set_title("Baseline")
ax[1].set_title("LSTM")
plt.tight_layout()
plt.show()

