import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../../Welding_Motoman/toolbox')
from angled_layers import SpeedHeightModel
import scienceplots


plt.style.use('science')
plt.rcParams['text.usetex'] = True

fig, ax= plt.subplots()
fig.set_size_inches(5,4)
fig.set_dpi(300)
marker_size = 2
plt_colors = [
    '#0C5DA5',
    '#00B945',
    '#FF9500',
    '#FF2C00',
]
plt_styles = [
    'solid',
    'dotted',
    'dashed',
    'dashdot'
]
marker_styles = [
    'o',
    '^',
    's',
    'D'
]
# labels = [
#         "Open-Loop Cold Model", 
#         "Open-Loop Hot Model",
#         "Closed-Loop Cold Model", 
#         "Closed-Loop Hot Model"
#         ]
labels = [
        'Layer Planning Only',
        'In-Process Correction'
        ]

model = SpeedHeightModel(a=-0.36997977, b=1.21532975)
v = np.linspace(3,17)
dh = model.v2dh(v)

ax.plot(v,dh)
ax.grid()
ax.set_xlabel(r"$v_T$ (mm/s)")
ax.set_ylabel(r"$\Delta h$ (mm)")
ax.set_title("160 IPM Deposition Model")
fig.savefig(f"deposition_model.png", dpi=fig.dpi)
plt.show()
