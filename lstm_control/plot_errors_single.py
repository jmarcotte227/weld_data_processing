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
    "process_error/2026_02_12_16_06_04_tube_lstm_control_layer_err.csv",
    delimiter=','
)
# rms error calc
fig, ax = plt.subplots(1,1)
rms_errors_1 = []
for layer in range(errors_1.shape[0]):
    rms_errors_1.append(rms_error(errors_1[layer,2:-2]))
    # rms_errors_1.append(rms_error(errors_1[layer,:]))
print(f"RMS 1: {rms_errors_1[-1]}")
ax.plot(rms_errors_1)

ax.legend([
    'Linearized LSTM Control',
    ''
])
ax.set_ylim([0,1.3])
ax.set_ylabel("RMSE (mm)")
ax.set_xlabel("Layer, No.")
plt.show()

fig,ax = plt.subplots(1,1)
ax.plot(-errors_1[-1,:])
# ax.set_ylim([-1,4])
plt.show()

min_e = errors_1.sum(axis=0).min()
max_e = errors_1.sum(axis=0).max()

# heatmap of errors
fig,ax=plt.subplots(1,2)
im_1 = ax[0].imshow(errors_1, cmap='inferno', vmin=min_e, vmax=max_e)
ax[0].invert_yaxis()
ax[0].set_title("Baseline")
ax[1].invert_yaxis()
ax[1].set_title("LSTM Controller")
fig.colorbar(im_1)
plt.tight_layout()
plt.show()

# bar chart of error sums for each segment
max_e = errors_1.sum(axis=0).max()
min_e = errors_1.sum(axis=0).min()
fig,ax = plt.subplots(2,1)
ax[0].plot(errors_1.sum(axis=0))
ax[0].set_ylim([min_e, max_e])
ax[1].set_ylim([min_e, max_e])
plt.show()
