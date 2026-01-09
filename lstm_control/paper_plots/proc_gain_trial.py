import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.metrics import rms

if __name__=="__main__":
    # test directory
    test_dir = "test_gains/20251231-102110/"
    # test_dir = "test_gains/20251113-231314/"
    # test_dir = "test_gains/20251113-174555/"
    # test_dir = "test_gains/20251113-160711/"
    # test_dir = "test_gains/20251113-170128/"
    # test_dir = "test_gains/20251113-141635/"
    # test_dir = "test_gains/20251015-153041/"
    # test_dir = "test_gains/20251016-140633/"
    test_data = torch.load(f"{test_dir}test_results.pt")

    error_results = test_data["results"]
    print(error_results.shape)
    vels = test_data["velocity"]
    b_vals = test_data["beta"]
    a_vals = test_data["alpha"]
    dh_vals = test_data["layer_dh"]

    # we want to aggregate the rms errors of each layer in all trials
    # for different values of alpha and beta
    ba_results_mean = np.zeros((error_results.shape[0], error_results.shape[1]))
    ba_results_std = np.zeros((error_results.shape[0], error_results.shape[1]))

    b_idx = 1
    a_idx = 1
    dh_idx = 1
    # a_idx = -1

    # for a_idx in range(len(a_vals)):

    fig, ax = plt.subplots(1,1)
    for trial in range(error_results.shape[3]):
    # for trial in :
        print(error_results[
              b_idx,
              a_idx,
              dh_idx,
              trial,
              0,
              0])
        rms_list = []
        for layer in range(error_results.shape[4]):
            rms_list.append(rms(error_results[
                                b_idx,
                                a_idx,
                                dh_idx,
                                trial,
                                layer,
                                :
                                ]))
        ax.plot(rms_list)
        print(np.std(rms_list[10:]))
        # print(f"Avg RMSE: last 50 layers: {np.mean(rms_list[75:])}")
        # now do summary statistics for the alpha and beta combo
    ba_results_mean[b_idx, a_idx] = np.mean(rms_list)
    ba_results_std[b_idx, a_idx] = np.std(rms_list)

    ax.set_title(f"RMSE: $\\alpha={a_vals[a_idx]}, \\beta={b_vals[b_idx]}, dh={dh_vals[dh_idx]}$")
    ax.set_xlabel("Layer No.")
    ax.set_ylabel("RMSE (mm)")
    
    fig.tight_layout()

    plt.show()

    fig,ax = plt.subplots(1,1)
    ax.hist(np.reshape(vels,(-1,1)), density=True)
    ax.set_title(f"RMSE: $\\alpha={a_vals[a_idx]}, \\beta={b_vals[b_idx]}, dh={dh_vals[dh_idx]}$")
    plt.show()


