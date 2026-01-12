import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

from matplotlib import rc

rc('text', usetex=True)
rc('font',**{'family':'sans-serif','sans-serif':['Latin Modern Sans']})

def rms(x):
    return np.sqrt(np.sum(np.square(x))/len(x))

def meas_ss(signal, threshold, window_size=5):
    '''
    finds the steady state of the signal, 
    '''
    signal = np.array(signal)
    sliding_std_dev = []
    num_windows = len(signal)-window_size+1
    
    for i in range(num_windows):
        window = signal[i: i+window_size]
        window_std = np.std(window)
        sliding_std_dev.append(window_std)

    last_unsteady_window_index = -np.inf
    for i in range(len(sliding_std_dev) - 1, -1, -1):
        if sliding_std_dev[i] < threshold:
            last_unsteady_window_index = i
            break # Found the last "bad" window

    ss_idx = last_unsteady_window_index + 1

    if ss_idx < 0:
        print(f"System did not reach Std. Dev. threshold: {threshold}")

        return False, (None, None, None)

    if ss_idx == 0:
        print(f"System within Std. Dev threshold from the start.")

    ss_mean = np.mean(signal[ss_idx:])
    ss_std = np.std(signal[ss_idx:])

    return True, (ss_mean, ss_std, ss_idx)

if __name__=="__main__":


    V_MIN = 0.33
    V_MAX = 1.17
    # test directory
    test_dir = "data/20260107-043810_gain_tests/"
    # test_dir = "data/20260107-105655_gain_tests/"
    # test_dir = "test_gains/20260102-150353/"
    # test_dir = "test_gains/20251231-102110/"
    # test_dir = "test_gains/20251113-231314/"
    # test_dir = "test_gains/20251113-174555/"
    # test_dir = "test_gains/20251113-170128/"
    # test_dir = "test_gains/20251009-195542/"
    # test_dir = "test_gains/20251015-153041/"
    # test_dir = "test_gains/20251016-140633/"
    test_data = torch.load(f"{test_dir}test_results.pt")

    error_results = test_data["results"]
    vels = test_data["velocity"]
    b_vals = test_data["beta"]
    a_vals = test_data["alpha"]
    dh_vals = test_data["layer_dh"]

    fig_all, ax_all = plt.subplots(2,3, sharex=True, sharey=True)
    fig_all.set_size_inches(6.4, 3.5)
    cbar_ax = fig_all.add_axes([.86, .1, .03, .8])
    ax_idx = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]
    for dh_idx in range(error_results.shape[2]-1):
        # we want to aggregate the rms errors of each layer in all trials
        # for different values of alpha and beta
        ba_results_mean = np.zeros((error_results.shape[0], error_results.shape[1]))
        ba_results_std = np.zeros((error_results.shape[0], error_results.shape[1]))
        ba_results_stab = np.zeros((error_results.shape[0], error_results.shape[1]))

        for b_idx in range(error_results.shape[0]):
            for a_idx in range(error_results.shape[1]):
                rms_list = []
                for trial in range(error_results.shape[3]):
                    # for layer in range(error_results.shape[4]):
                    #     rms_list.append(rms(error_results[
                    #                         b_idx,
                    #                         a_idx,
                    #                         dh_idx,
                    #                         trial,
                    #                         layer,
                    #                         :
                    #                         ]))
                    rms_list.append(rms(error_results[
                                        b_idx,
                                        a_idx,
                                        dh_idx,
                                        trial,
                                        -1,
                                        :
                                        ]))
                # # check if the system is stable
                # ret, stats = meas_ss(rms_list, 0.1)
                # if ret:
                #     ba_results_stab[b_idx, a_idx] = stats[2]
                #     # now do summary statistics for the alpha and beta combo
                #     ba_results_mean[b_idx, a_idx] = stats[0]
                #     ba_results_std[b_idx, a_idx] = stats[1]
                #     print(stats[2])
                # else:
                #     ba_results_stab[b_idx, a_idx] = np.nan
                #     ba_results_mean[b_idx, a_idx] = np.nan
                #     ba_results_std[b_idx, a_idx] = np.nan
                # print(len(rms_list))
                ba_results_mean[b_idx, a_idx]=np.mean(rms_list)
                ba_results_std[b_idx, a_idx]=np.std(rms_list)

        a_vals_str = ["${:.2f}$".format(x) for x in a_vals]
        b_vals_str = ["${:.2f}$".format(x) for x in b_vals]
        
        print(np.min(ba_results_mean))
        print(np.max(ba_results_mean))
        sns.heatmap(
            ba_results_mean,
            ax=ax_all[ax_idx[dh_idx]],
            xticklabels=a_vals_str,
            yticklabels=b_vals_str,
            vmax=V_MAX,
            vmin=V_MIN,
            cbar_ax=None if dh_idx!=1 else cbar_ax,
            cbar = dh_idx==1,
            cbar_kws=None if dh_idx!=1 else {'label': 'RMSE (mm)'},
            cmap='viridis'
            # norm=LogNorm()
        )

        ax_all[ax_idx[dh_idx]].set_xticks(np.arange(len(a_vals_str))[::3] + 0.5)
        ax_all[ax_idx[dh_idx]].set_yticks(np.arange(len(b_vals_str))[::3] + 0.5)
        ax_all[ax_idx[dh_idx]].set_title("$\\Delta H_{nom}$="+ f"${dh_vals[dh_idx]:.1f}$")
        ax_all[ax_idx[dh_idx]].set_xlabel(r"$\alpha$")
        ax_all[ax_idx[dh_idx]].set_ylabel(r"$\beta$")
        ax_all[ax_idx[dh_idx]].set_aspect("equal")
    # fig_all.suptitle("Mean RMSE - Gain Analysis")
    fig_all.tight_layout(rect=[0, 0, .85, 1])
    # fig_all.tight_layout()

    plt.savefig("output_plots/gain_test.png", dpi=300)
    plt.savefig("output_plots/gain_test.tiff", dpi=300)
    plt.show()
        # plt.plot(a_vals, ba_results_mean[0, :])
        # plt.show()
