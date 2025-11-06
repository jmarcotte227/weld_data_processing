import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def rms_error(data):
    data = np.array(data)
    n = 0
    num = 0
    for i in data:
        if not np.isnan(i): 
            num = num + i**2
            n+=1
    return np.sqrt(num/n)

errors_1 = np.loadtxt(
    # "process_error/wall_lstm_control_2025_10_31_14_30_40_layer_err.csv",
    "process_error/wall_lstm_baseline_control_2025_11_05_12_38_13_layer_err.csv",
    delimiter=','
)

errors_2 = np.loadtxt(
    "process_error/wall_lstm_control_2025_11_05_13_17_59_layer_err.csv",
    # "process_error/wall_lstm_control_2025_10_31_13_34_50_layer_err.csv",
    delimiter=','
)

# for layer in range(4,errors.shape[0]):
#     plt.plot(errors[layer,2:-2])
# plt.show()

# rms error calc
fig, ax = plt.subplots(1,1)
rms_errors_1 = []
for layer in range(25, errors_1.shape[0]):
    rms_errors_1.append(rms_error(errors_1[layer,2:-2]))
    # rms_errors_1.append(rms_error(errors_1[layer,:]))
ax.plot(rms_errors_1)
rms_errors = []
for layer in range(25, errors_2.shape[0]):
    rms_errors.append(rms_error(errors_2[layer,2:-2]))
    # rms_errors.append(rms_error(errors_2[layer,:]))
ax.plot(rms_errors)

ax.legend([
    # r"$\alpha=1.0, \beta=0.2$",
    # r"$\alpha=1.4, \beta=0.1$",
    'Baseline: Log-Log Control',
    'Linearized LSTM Control',
    ''
])
ax.set_ylim([0,1.3])
ax.set_ylabel("RMSE (mm)")
ax.set_xlabel("Layer, No.")
plt.show()


# trim the first few layers
errors_1 = errors_1[25:,:]
errors_2 = errors_2[25:, :]
max_e = max(errors_1.max(), errors_2.max())
min_e = min(errors_1.min(), errors_2.min())

# heatmap of errors
fig,ax=plt.subplots(1,2)
im_1 = ax[0].imshow(errors_1, cmap='inferno', vmin=min_e, vmax=max_e)
im_2 = ax[1].imshow(errors_2, cmap='inferno', vmin=min_e, vmax=max_e)
ax[0].invert_yaxis()
ax[0].set_title("Baseline")
ax[1].invert_yaxis()
ax[1].set_title("LSTM Controller")
fig.colorbar(im_1)
fig.colorbar(im_2)
plt.tight_layout()
plt.show()

# bar chart of error sums for each segment
max_e = max(errors_1.sum(axis=0).max(), errors_2.sum(axis=0).max())
min_e = min(errors_1.sum(axis=0).min(), errors_2.sum(axis=0).min())
fig,ax = plt.subplots(2,1)
ax[0].plot(errors_1.sum(axis=0))
ax[1].plot(errors_2.sum(axis=0))
ax[0].set_ylim([min_e, max_e])
ax[1].set_ylim([min_e, max_e])
plt.show()
