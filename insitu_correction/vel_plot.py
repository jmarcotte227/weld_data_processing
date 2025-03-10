import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


for layer in range (101,102):
    try:
        cor_vel_data = np.loadtxt(f'../../recorded_data/ER4043_bent_tube_large_hot_streaming_2025_03_06_feedback_troubleshooting/layer_{layer}/v_plan.csv', delimiter=',')
        vel_data = np.loadtxt(f'../../recorded_data/ER4043_bent_tube_large_hot_streaming_2025_03_06_feedback_troubleshooting/layer_{layer}/v_cmd.csv', delimiter=',')

        time_step = np.linspace(0,len(vel_data)/125, len(vel_data))
        plt.plot(time_step, vel_data, label="Planned")
        plt.plot(time_step, cor_vel_data, label="With Correction")
    except:
        pass
plt.xlabel("Time (s)")
plt.ylabel("Velocity (mm/s)")
plt.title("Layer 101 Velocity")
plt.legend()
plt.show()

plt.plot(time_step[1000:-1000], (cor_vel_data-vel_data)[1000:-1000])
plt.xlabel("Time (s)")
plt.ylabel("Error (mm)")
plt.show()

data = (cor_vel_data-vel_data)[1000:-1000]
data = data-np.mean(data)
t=time_step[1000:-1000]

N = len(data)
T = 1/125

data_f=fft(data)
tf = fftfreq(N,T)[:N//2]
plt.plot(tf, 2.0/N * np.abs(data_f[0:N//2]))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()


