import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
from cycler import cycler

def main():

    PROC_DATA_DIR = "../data/"

    DATA_1 = "2026_01_12_11_48_35_wall_lstm_baseline_control"
    DATA_2 = "2026_01_12_10_21_38_wall_lstm_control"

    NUM_LAYERS = 100

    H_1 = np.loadtxt(f"{PROC_DATA_DIR}/calc_h/{DATA_1}_h.csv", delimiter=',')
    H_2 = np.loadtxt(f"{PROC_DATA_DIR}/calc_h/{DATA_2}_h.csv", delimiter=',')

    rc('text', usetex=True)
    rc('font',**{'family':'sans-serif','sans-serif':['Latin Modern Sans']})

    # colors = ["#7fc97f","#beaed4","#fdc086"]
    my_colors = [
        '#009e73',
        '#0072b2',
        '#f0e442',
        '#e69f00',
        '#d55e00',
        '#cc79a7',
    ]

    # Set the global color cycle
    plt.rc('axes', prop_cycle=cycler(color=my_colors))

    datasets = [H_1, H_2]

    x_coord = np.linspace((-23+0.5)*np.pi, (23-0.5)*np.pi, 46)
    print(x_coord)
    
    for idx, d_set in enumerate(datasets):
        fig, ax = plt.subplots(1, 1)
        ax.plot([x_coord[0], x_coord[-1]], [163, 163], 'r:')

        for layer in range(NUM_LAYERS):
            if layer%2:
                ax.plot(x_coord, d_set[layer])
            else:
                ax.plot(x_coord, np.flip(d_set[layer]))

        ax.set_aspect("equal")
        ax.set_xlabel("$X$")
        ax.set_ylabel("$Z$")
        ax.set_ylim([0,170])
        ax.grid()
        ax.spines[['right', 'top']].set_visible(False)
        fig.set_size_inches(6.4, 6.4*1.05)
        fig.tight_layout()
    # fig.subplots_adjust(left=-0.11)
        if idx==0:
            plt.savefig(f"output_plots/exp_wall_vis_ll.png", dpi=300)
        else:
            plt.savefig(f"output_plots/exp_wall_vis_lstm.png", dpi=300)
        plt.show()
        
if __name__=="__main__":
    main()
