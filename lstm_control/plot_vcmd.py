import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../../Welding_Motoman/toolbox')
from angled_layers import avg_by_line

# REC_DIR = "../../../recorded_data/wall_lstm_control_2025_10_31_14_30_40/"
# REC_DIR = "../../recorded_data/wall_lstm_control_2025_10_31_13_34_50/"
REC_DIR = "../../recorded_data/wall_lstm_control_2025_11_05_13_17_59/"

v_cmds_cum = []
LAYERS = [9,24,35,40]
fig, ax = plt.subplots(3,1, sharex=True)
for layer in LAYERS: 
    v_cmd = np.loadtxt(f"{REC_DIR}layer_{layer}/v_cmd.csv", delimiter=",")

    js_exe = np.loadtxt(f"{REC_DIR}layer_{layer}/weld_js_cmd.csv", delimiter=",")

    job_no = np.linspace(0,48, 49)

    v_cmds = []
    for num in job_no:
        idx = np.where(js_exe[:,1]==num)[0][0]
        v_cmds.append(v_cmd[idx+1])
        v_cmds_cum.append(v_cmd[idx+1])

    ax[0].plot(v_cmds)
ax[0].legend(LAYERS)
ax[0].set_ylabel(r"$v_{cmd} (\alpha=1.0, \beta=0.1)$ (mm/s)")

# REC_DIR = "../../recorded_data/wall_lstm_control_2025_10_31_14_30_40/"
# REC_DIR = "../../recorded_data/wall_lstm_control_2025_11_05_13_17_59/"
for layer in LAYERS: 
    dh_d= np.loadtxt(f"{REC_DIR}layer_{layer}/dh_d.csv", delimiter=",")

    js_exe = np.loadtxt(f"{REC_DIR}layer_{layer}/weld_js_cmd.csv", delimiter=",")

    job_no = np.linspace(0,48, 49)

    ax[1].plot(dh_d)
ax[1].legend(LAYERS)
ax[1].set_ylabel(r"$\Delta H_d (mm)$")
ax[1].set_ylim([0,4])

for layer in LAYERS: 
    dh_prev= np.loadtxt(f"{REC_DIR}layer_{layer}/dh_prev_all.csv", delimiter=",")

    js_exe = np.loadtxt(f"{REC_DIR}layer_{layer}/weld_js_cmd.csv", delimiter=",")

    job_no = np.linspace(0,47, 48)

    ax[2].plot(dh_prev)

ax[2].set_xlabel("Segment Index")
ax[2].legend(LAYERS)
ax[2].set_ylim([0,4])
ax[2].set_ylabel(r"$\Delta H(mm)$")

plt.show()

# plot histogram from these tests vs the previous tests
fig, ax = plt.subplots(2,1, sharex=True)
# REC_DIR = "../../recorded_data/wall_lstm_control_2025_10_31_13_34_50/"
REC_DIR = "../../recorded_data/wall_lstm_control_2025_11_05_13_17_59/"
v_cmds = []
for layer in range(50): 
    v_cmd = np.loadtxt(f"{REC_DIR}layer_{layer}/v_cmd.csv", delimiter=",")

    js_exe = np.loadtxt(f"{REC_DIR}layer_{layer}/weld_js_cmd.csv", delimiter=",")

    job_no = np.linspace(0,48, 49)

    for num in job_no:
        idx = np.where(js_exe[:,1]==num)[0][0]
        v_cmds.append(v_cmd[idx+1])

fig,ax = plt.subplots(1,1)
ax.hist(v_cmds)
ax.set_xlabel("v_cmd (mm/s)")
ax.set_ylabel("Counts Scaled")
plt.show()

fig,ax = plt.subplots(2,1)
ax[0].hist(v_cmds)
# REC_DIR = "../../recorded_data/ER4043_bent_tube_large_hot_2024_11_06_12_27_19/"
# REC_DIR = "../../recorded_data/wall_lstm_control_2025_11_05_13_17_59/"
# v_cmds = []
# for layer in range(1,105): 
#     v_cmd = np.loadtxt(f"{REC_DIR}layer_{layer}/velocity_profile.csv", delimiter=",")

#     for j in range(len(v_cmd)):
#         v_cmds.append(v_cmd[j])

# ax[1].hist(v_cmds)
# ax[1].set_xlabel("v_cmd (mm/s)")
# ax[0].set_ylabel("Experiment counts")
# ax[1].set_ylabel("Training counts")


# plt.show()





