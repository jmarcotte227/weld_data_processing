import numpy as np
import matplotlib.pyplot as plt

for layer in range (70,105):
    try:
        error_data = np.loadtxt(f'../../recorded_data/ER4043_bent_tube_large_hot_streaming_2025_03_06_feedback_troubleshooting/layer_{layer}/error.csv', delimiter=',')
        error_data = error_data[200:-200]
        plt.plot(error_data)
        e_sum=0
        for e in error_data:
            e_sum+=e**2
        rmse = np.sqrt(e_sum/len(error_data))
        print(f"Layer {layer} RMSE: ", rmse)
    except:
        pass
plt.show()
