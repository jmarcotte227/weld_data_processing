import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib import rc

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

# colors = ["#7fc97f","#beaed4","#fdc086"]
colors = ["#66c2a5","#fc8d62","#8da0cb","#ffff99", "#386cb0", "#f0027f"]

test_data_2 = torch.load(f"data/Linearized_QP_Control_plantfb_False_pmodel_h-8_part-0_loss-0.4866_cmodel_h-8_part-1_loss-0.041120251218-233750/test_results.pt")

times = [x*1000 for x in test_data_2["step_times"]] # convert to ms
print(f"Max Time: {np.max(times)}")
bins=10

fig,ax = plt.subplots(1,1)

ax.hist(
    times,
    bins=bins,
    density=True,
    color=colors[4],
    rwidth=0.95
)
ax.spines[['right', 'top']].set_visible(False)
ax.set_ylabel("Density")
ax.set_xlabel("Execution Time (ms)")
    
ax.set_title("LSTM MPC Execution Time")
fig.tight_layout()
plt.savefig("output_plots/exe_time.png", dpi=300)

plt.show()
