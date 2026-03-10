import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from statsmodels.graphics import tsaplots


# REC_DIR = "../../recorded_data/2026_01_12_11_48_35_wall_lstm_baseline_control/"

# REC_DIR = "../../recorded_data/2026_01_12_10_21_38_wall_lstm_control/"
# dhs = np.loadtxt(
#     "data/calc_dh/2026_01_12_10_21_38_wall_lstm_control_dh.csv",
#     delimiter=','
# )

REC_DIR = "../../recorded_data/2026_02_23_11_23_56_tube_lstm_control/"
dhs = np.loadtxt(
    "data/calc_dh/2026_02_23_11_23_56_tube_lstm_control_dh.csv",
    delimiter=','
)

errors=np.zeros((100,46))
fig, ax = plt.subplots(2, 5)
layers = [1, 24, 49, 74, 99]
for i, layer in enumerate(layers):
    lstm_pred = np.loadtxt(
        f"{REC_DIR}layer_{layer}/lstm_pred.csv",
        delimiter=','
    )
    dh = dhs[layer]
    if layer == 49:
        print(lstm_pred)
        print(dh)
    for j in range(46):
        errors[layer,j] = (dh[j]-lstm_pred[j])

    tsaplots.plot_acf(dh, ax = ax[0,i], title=f"Layer {layer} dH", missing='drop')
    tsaplots.plot_acf(errors[layer], ax = ax[1,i], title=f"Layer {layer} Error", missing='drop')
    # fig = tsaplots.plot_acf(errors[layer])
    # fig.suptitle(f"Error Auto-correlation - Layer {layer}")
ax[0,0].set_ylabel("dH")
ax[1,0].set_ylabel("Prediction Error")
fig.tight_layout()
plt.show()


REC_DIR
REC_DIR = "../../recorded_data/2026_01_12_11_48_35_wall_lstm_baseline_control/"

# fig,ax=plt.subplots(1,1)
# ax.hist(errors.reshape(-1,1), bins=300, density=True)
# ax.set_xlim([-3, 3])
# ax.set_xlabel("Error (mm)")
# ax.set_ylabel("Counts")
# plt.show()
