import numpy as np
import matplotlib.pyplot as plt



# REC_DIR = "../../recorded_data/wall_lstm_baseline_control_2025_11_05_12_38_13/"
# REC_DIR = "../../recorded_data/wall_lstm_control_2025_11_05_13_17_59/"
# REC_DIR = "../../../recorded_data/wall_lstm_control_2025_10_31_14_30_40/"
# REC_DIR = "../../recorded_data/wall_lstm_control_2025_10_31_13_34_50/"

# REC_DIR = "../../../recorded_data/wall_lstm_control_2025_10_31_14_30_40/"
REC_DIR = "../../recorded_data/2026_02_23_11_23_56_tube_lstm_control/"


errors=np.zeros((100,46))
for layer in range(100):
    lstm_pred = np.loadtxt(
        f"{REC_DIR}layer_{layer}/lstm_pred.csv",
        delimiter=','
    )
    dh_prev = np.loadtxt(
        f"{REC_DIR}layer_{layer}/dh_prev_all.csv",
        delimiter=','
    )
    print(dh_prev)
    for j in range(45):
        errors[layer,j] = (dh_prev[j]-lstm_pred[j])

fig,ax=plt.subplots(1,1)
ax.hist(errors.reshape(-1,1), bins=300, density=True)
ax.set_xlim([-3, 3])
ax.set_xlabel("Error (mm)")
ax.set_ylabel("Counts")
plt.show()

bins = np.linspace(-4,4,20)
# for i in range(50):
#     fig, ax = plt.subplots(1,1)
#     ax.hist(errors[:,i], bins=bins, density=True)
#     ax.set_xlim([-4,4])
#     plt.show()

# plot a few example layers
LAYERS = [9, 24, 40]
fig,ax = plt.subplots()
for layer in LAYERS:
    lstm_pred = np.loadtxt(
        f"{REC_DIR}layer_{layer}/lstm_pred.csv",
        delimiter=','
    )
    dh_prev = np.loadtxt(
        f"{REC_DIR}layer_{layer}/dh_prev_all.csv",
        delimiter=','
    )
    ax.plot(dh_prev-lstm_pred[:-1])

ax.legend(LAYERS)
ax.set_ylim([0,2.2])
ax.set_xlabel("Segment Index")
ax.set_ylabel("Error (mm)")
plt.show()
