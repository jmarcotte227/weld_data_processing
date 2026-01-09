import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

# colors = ["#7fc97f","#beaed4","#fdc086"]
colors = ["#66c2a5","#fc8d62","#8da0cb","#ffff99", "#386cb0", "#f0027f"]
# used in simulation
# cont_model_dir = "models/model_h-8_part-1_loss-0.0684/"
# cont_data = torch.load(f"{cont_model_dir}/model_h-8_part-1_loss-0.0684.pt")
PLANT = "model_h-8_part-1_loss-0.4868"
CONT = "model_h-8_part-1_loss-0.0835"

# opposites
cont_model_dir = "models/model_h-8_part-0_loss-0.0533/"
cont_data = torch.load(f"{cont_model_dir}/model_h-8_part-0_loss-0.0533.pt")
# plant_model_dir = "models/model_h-8_part-1_loss-0.3931/"
# plant_data = torch.load(f"{plant_model_dir}/model_h-8_part-1_loss-0.3931.pt")

plant_model_dir = f"models/{PLANT}/"
plant_data = torch.load(f"{plant_model_dir}/{PLANT}.pt")
cont_model_dir = f"models/{CONT}/"
cont_data = torch.load(f"{cont_model_dir}/{CONT}.pt")

# load the loss data
plant_train = np.loadtxt(f"{plant_model_dir}train_loss.csv", delimiter=',')
plant_valid = np.loadtxt(f"{plant_model_dir}valid_loss.csv", delimiter=',')
cont_train = np.loadtxt(f"{cont_model_dir}train_loss.csv", delimiter=',')
cont_valid = np.loadtxt(f"{cont_model_dir}valid_loss.csv", delimiter=',')

# min_e = np.argmin(plant_valid)
# print(min_e)
# print(plant_train[min_e])
# print(plant_valid[min_e])
# min_e = np.argmin(cont_valid)
# print(min_e)
# print(cont_train[min_e])
# print(cont_valid[min_e])
# print(min(cont_valid))

print("---Plant---")
print(f"LR: {plant_data['lr']}")
print(f"WD: {plant_data['wd']}")
print(f"Epoch: {plant_data['epoch']}")
print(f"Test: {plant_data['test_loss']}")
print(f"Val: {plant_data['val_loss']}")
min_e = np.argmin(plant_valid)
print(f"Val: ", plant_valid[min_e])
print(f"Train: ", plant_train[min_e])
print("Epoch: ", min_e)

print("---Control---")
print(f"LR: {cont_data['lr']}")
print(f"WD: {cont_data['wd']}")
print(f"Epoch: {cont_data['epoch']}")
print(f"Test: {cont_data['test_loss']}")
print(f"Val: {cont_data['val_loss']}")
min_e = np.argmin(cont_valid)
print(f"Val: ", cont_valid[min_e])
print(f"Train: ", cont_train[min_e])
print("Epoch: ", min_e)

fig,ax = plt.subplots(2,1)
ax[1].plot(cont_train, color=colors[5])
ax[1].plot(cont_valid, color=colors[4])
ax[0].plot(plant_train, color=colors[5])
ax[0].plot(plant_valid, color=colors[4])
ax[0].legend(["Training", "Validation"])
ax[1].legend(["Training", "Validation"])
ax[0].set_title("Conventional LSTM Loss")
ax[1].set_title("Innovation Driven LSTM Loss")

for a in ax:
    a.spines[['right', 'top']].set_visible(False)
    a.set_ylabel("Mean-Squared Error Loss")
    a.set_xlabel("Epoch")
    a.set_ylim([0,1.2])
    
fig.tight_layout()
plt.savefig("output_plots/loss.png", dpi=300)

plt.show()
