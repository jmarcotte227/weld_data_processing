import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import scienceplots

plt.style.use('science')
# colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# print('\n'.join(color for color in colors))
# exit()
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble']=r'\usepackage{bm}'

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


for layer in [101,103]:
    print("Layer: ",layer)
    fig, [ax1,ax2] = plt.subplots(2,1, sharex=True)
    fig.set_size_inches(5,4)
    fig.set_dpi(300)
    try:
        cor_vel_data = np.loadtxt(f'../../recorded_data/ER4043_bent_tube_large_hot_streaming_2025_03_06_feedback_troubleshooting/layer_{layer}/v_plan.csv', delimiter=',')
        vel_data = np.loadtxt(f'../../recorded_data/ER4043_bent_tube_large_hot_streaming_2025_03_06_feedback_troubleshooting/layer_{layer}/v_cmd.csv', delimiter=',')
        error_data = np.loadtxt(f'../../recorded_data/ER4043_bent_tube_large_hot_streaming_2025_03_06_feedback_troubleshooting/layer_{layer}/error.csv', delimiter=',')

        time_step = np.linspace(0,len(vel_data)/125, len(vel_data))
        time_step_e = np.linspace(0,len(error_data)/30, len(error_data))
        ax1.plot(time_step, vel_data, label=r"$\bar{\bm{v}}_{T,"+f"{layer-82}"+r"}$", c=plt_colors[2])
        ax1.plot(time_step, cor_vel_data, label=r"$\bm v_{T,"+f"{{{layer-82}}}"+r"}$",c=plt_colors[0])
        ax2.plot(time_step_e, error_data, label=r"$\bm e_"+f"{{{layer-82}}}"+r"$", c=plt_colors[0])
    except Exception as e:
        print(e)
    ax2.set_xlabel("Time (s)")
    ax1.set_ylabel("Velocity (mm/s)")
    ax2.set_ylabel("Error (mm)")
    ax1.set_title(f"Layer {layer-82} Velocity")
    ax1.legend(facecolor='white', 
           framealpha=0.8,
           frameon=True,
           # loc='lower center',
           # ncol=2,
           # bbox_to_anchor=(0.5,-0.8)
           )
    ax2.legend(facecolor='white', 
           framealpha=0.8,
           frameon=True,
           # loc='lower center',
           # ncol=2,
           # bbox_to_anchor=(0.5,-0.8)
           )
    ax1.set_ylim(2.2, 8.5)
    ax2.set_ylim(-2,0.6)
    ax2.grid()
    ax1.grid()
    fig.savefig(f"velocity_plot_{layer-82}.png", dpi=fig.dpi)
    plt.show()
# idx_start = 200
# idx_end = -200
# plt.plot(time_step[idx_start:idx_end], (cor_vel_data-vel_data)[idx_start:idx_end])
# plt.xlabel("Time (s)")
# plt.ylabel("Error (mm)")
# plt.title(f"Layer {layer} Error")
# plt.show()

# data = (cor_vel_data-vel_data)[1000:-1000]
# data = data-np.mean(data)
# t=time_step[1000:-1000]

# N = len(data)
# T = 1/125

# data_f=fft(data)
# tf = fftfreq(N,T)[:N//2]
# plt.plot(tf, 2.0/N * np.abs(data_f[0:N//2]))
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Amplitude")
# plt.grid()
# plt.show()


