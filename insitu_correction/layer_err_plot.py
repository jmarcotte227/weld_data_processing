import numpy as np
import matplotlib.pyplot as plt

layer = 95
layer_errors = np.loadtxt("error_data/ER4043_bent_tube_large_hot_streaming_2025_03_06_feedback_troubleshooting_layer_err.csv", delimiter=',')

plt.plot(layer_errors[layer,:])
plt.show()
