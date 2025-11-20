import numpy as np
import glob

def main():
    REC_DIR = "../../../recorded_data/"
    # DATASET = "2025_11_19_11_50_06_AL_WLJ_dataset0"
    # DATASET = "2025_11_19_12_20_00_AL_WLJ_dataset1"
    # DATASET = "2025_11_19_12_50_06_AL_WLJ_dataset2"
    DATASET = "2025_11_19_13_19_53_AL_WLJ_dataset3"

    layer_dirs = glob.glob(f"{REC_DIR}{DATASET}/layer_*")
    num_layer = len(layer_dirs)
    v_cmd_list = np.zeros((num_layer, 50))

    for layer in range(num_layer):
        v_cmd_all = np.loadtxt(
            f"{REC_DIR}{DATASET}/layer_{layer}/v_cmd.csv",
            delimiter=','
        )
        v_cmd_idx = np.loadtxt(
            f"{REC_DIR}{DATASET}/layer_{layer}/v_cor_idx.csv",
            delimiter=','
        )

        idxs = np.linspace(0,49,50, dtype=int)

        for idx in idxs:
            loc = np.where(v_cmd_idx==idx)[0][0]
            v_cmd_list[layer,idx] = v_cmd_all[loc]

    np.savetxt(f"v_set/{DATASET}_v_cmd.csv", v_cmd_list, delimiter=',')

if __name__=="__main__":
    main()
