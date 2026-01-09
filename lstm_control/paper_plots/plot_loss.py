import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

# colors = ["#7fc97f","#beaed4","#fdc086"]
colors = ["#66c2a5","#fc8d62","#8da0cb","#ffff99", "#386cb0", "#f0027f"]
cont_model_dir = "models/model_h-8_part-1_loss-0.0411/"
plant_model_dir = "models/model_h-8_part-0_loss-0.4866/"

# load the loss data
plant_train = np.loadtxt(f"{plant_model_dir}train_loss.csv", delimiter=',')
plant_valid = np.loadtxt(f"{plant_model_dir}valid_loss.csv", delimiter=',')
cont_train = np.loadtxt(f"{cont_model_dir}train_loss.csv", delimiter=',')
cont_valid = np.loadtxt(f"{cont_model_dir}valid_loss.csv", delimiter=',')

min_e = np.argmin(plant_valid)
print(min_e)
print(plant_train[min_e])
print(plant_valid[min_e])
min_e = np.argmin(cont_valid)
print(min_e)
print(cont_train[min_e])
print(cont_valid[min_e])

fig,ax = plt.subplots(2,1)
ax[0].plot(cont_train, color=colors[5])
ax[0].plot(cont_valid, color=colors[4])
ax[1].plot(plant_train, color=colors[5])
ax[1].plot(plant_valid, color=colors[4])
ax[0].legend(["Training", "Validation"])
ax[1].legend(["Training", "Validation"])
ax[0].set_title("Control Model Loss")
ax[1].set_title("Plant Model Loss")

for a in ax:
    a.spines[['right', 'top']].set_visible(False)
    a.set_ylabel("Mean-Squared Error Loss")
    a.set_xlabel("Epoch")
    
fig.tight_layout()
plt.savefig("output_plots/loss.png", dpi=300)

plt.show()
