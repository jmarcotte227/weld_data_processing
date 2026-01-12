import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.stats import laplace
import seaborn as sns

save_name = "noise_fit.png"

rc('text', usetex=True)
rc('font',**{'family':'sans-serif','sans-serif':['Latin Modern Sans']})

# colors = ["#7fc97f","#beaed4","#fdc086"]
# colors = ["#66c2a5","#fc8d62","#8da0cb","#ffff99", "#386cb0", "#f0027f"]
colors = [
    '#56b4e9',
    '#d55e00',
]

DATA_DIR = "data/"
# load baseline test set
def rms(x):
    return np.sqrt(np.sum(np.square(x))/len(x))

### Load Error Data
PLANT_MODEL = "model_h-8_part-0_loss-0.2000"
errors = np.loadtxt(f"data/noise_{PLANT_MODEL}/errors.csv", delimiter=',')
coeffs = np.loadtxt(f"data/noise_{PLANT_MODEL}/lap_coeffs.csv", delimiter=',')
loc = coeffs[0]
scale = coeffs[1]


fig,ax = plt.subplots()
ax.hist(
    errors,
    bins=100,
    density=True,
    color=colors[0]
)

# plot normal distribution over this
mu = np.mean(errors)
sigma = np.std(errors)
x = np.linspace(-3,3, 1000)
# plt.plot(x,stats.norm.pdf(x,mu,sigma))
# plt.plot(x,stats.norm.pdf(x,mu,sigma))
# plt.hist(laplace.rvs(loc=fitted_loc, scale=fitted_scale, size=len(errors)),density=True, bins=100)
ax.plot(
    x,
    laplace.pdf(
        x,
        loc=loc,
        scale=scale,
    ),
    color=colors[1]
)
# labels
ax.spines[['right', 'top']].set_visible(False)
ax.set_ylabel("Density")
ax.set_xlabel("Predicition Error (mm)")
ax.legend(["Laplace Distribution Fit","Prediction Error"])

fig.set_size_inches(6.4, 3)
# fig.suptitle("Noise", fontsize=20)
fig.tight_layout()
plt.savefig(f"output_plots/{save_name}", dpi=300)
plt.show()
