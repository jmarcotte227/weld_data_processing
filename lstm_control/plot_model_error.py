import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../../Welding_Motoman/toolbox/")

from angled_layers import SpeedHeightModel

dh_1 = np.loadtxt(
    "data/calc_dh/2026_02_25_09_57_16_tube_baseline_control_dh.csv",
    delimiter=','
)

dh_2 = np.loadtxt(
    "data/calc_dh/2026_02_23_11_23_56_tube_lstm_control_dh.csv",
    delimiter=','
)

baseline_v_set = np.loadtxt(
    "data/v_set/2026_02_25_09_57_16_tube_baseline_control_v_cmd.csv",
    delimiter=','
)


model = SpeedHeightModel(a=-0.4733,b=1.1747)

def rms_error(data):
    data = np.array(data)
    n = 0
    num = 0
    for i in data:
        if not np.isnan(i): 
            num = num + i**2
            n+=1
    return np.sqrt(num/n)

errors_lstm = []

for layer in range(1,dh_2.shape[0]):
    # load lstm prediction
    lstm_pred = np.loadtxt(
        f"../../recorded_data/2026_02_23_11_23_56_tube_lstm_control/layer_{layer}/lstm_pred.csv",
        delimiter=','
    )

    errors_lstm.append(rms_error((lstm_pred-dh_2[layer])[2:-2]))

errors_ll = []
for layer in range(1,dh_1.shape[0]):
    ll_pred = model.v2dh(baseline_v_set[layer])
    errors_ll.append(rms_error((ll_pred-dh_1[layer])[2:-2]))

fig,ax = plt.subplots(1,1)
ax.set_title("Prediction Error Trimmed")
ax.plot(errors_ll)
ax.plot(errors_lstm)
ax.legend([
    # r"$\alpha=1.0, \beta=0.2$",
    # r"$\alpha=1.4, \beta=0.1$",
    'Log-Log',
    'LSTM',
])
ax.set_ylim([0,2.2])
ax.set_ylabel(r"RMSE ($\Delta H -\Delta \hat H$)")
ax.set_xlabel(r"Layer No.")
plt.show()

