import numpy as np
import matplotlib.pyplot as plt

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
    "process_error/wall_lstm_control_2025_10_31_13_34_50_layer_err.csv",
    delimiter=','
)

errors_2 = np.loadtxt(
    "process_error/wall_lstm_control_2025_10_31_14_30_40_layer_err.csv",
    # "process_error/wall_lstm_control_2025_10_31_13_34_50_layer_err.csv",
    delimiter=','
)

# for layer in range(4,errors.shape[0]):
#     plt.plot(errors[layer,2:-2])
# plt.show()

# rms error calc
fig, ax = plt.subplots(1,1)
rms_errors_1 = []
for layer in range(errors_1.shape[0]):
    # rms_errors_1.append(rms_error(errors_1[layer,2:-2]))
    rms_errors_1.append(rms_error(errors_1[layer,:]))
ax.plot(rms_errors_1)
rms_errors = []
for layer in range(errors_2.shape[0]):
    # rms_errors.append(rms_error(errors_2[layer,2:-2]))
    rms_errors.append(rms_error(errors_2[layer,:]))
ax.plot(rms_errors)

ax.legend([
    r"$\alpha=1.0, \beta=0.2$",
    r"$\alpha=1.4, \beta=0.1$",
])
ax.set_ylim([0,1.5])
ax.set_ylabel("RMSE (mm)")
ax.set_xlabel("Layer, No.")
plt.show()

