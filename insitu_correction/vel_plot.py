import numpy as np
import matplotlib.pyplot as plt


for layer in range (71,72):
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
plt.title("Layer 91 Velocity")
plt.legend()
plt.show()
