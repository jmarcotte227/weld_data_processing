import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.io import savemat

TRIM=1
RAW_DATA_PATH = '../../../recorded_data'
PROC_DATA_PATH = ''
PART_NAMES = [
        "aprbs_1",
        "aprbs_2",
        "aprbs_3",
        "aprbs_4",
        ]
TEST_IDS = [
        "2025_11_19_11_50_06_AL_WLJ_dataset0",
        "2025_11_19_12_20_00_AL_WLJ_dataset1",
        "2025_11_19_12_50_06_AL_WLJ_dataset2",
        "2025_11_19_13_19_53_AL_WLJ_dataset3"
        ]

set_dict = {}
for idx, TEST_ID in enumerate(TEST_IDS):
    # load processed data (since it is compiled per part) 
    dh_data = np.loadtxt(f'calc_dh/{TEST_ID}_dh.csv', delimiter=',')
    # vel_calc = np.loadtxt(f'calc_v/{TEST_ID}_vel_calc.csv', delimiter=',')
    vel_set = np.loadtxt(f'v_set/{TEST_ID}_v_cmd.csv', delimiter=',')
    # temp = np.loadtxt(f'{PROC_DATA_PATH}/temp_data/{TEST_ID}_temps.csv', delimiter=',')
    # with open(f'{PROC_DATA_PATH}temp_data/{TEST_ID}_temps.pkl', 'rb') as file:
    #     temp = pickle.load(file)
    layer_count = dh_data.shape[0]
    part_dict = {}
    fig,ax = plt.subplots(1,1)
    # ax.plot(vel_calc[2,:])
    ax.step(np.linspace(0,49-2*TRIM,50-2*TRIM),vel_set[2,:])
    plt.show()

    for layer in range(0,layer_count):
        # find layer direction
        direction = np.loadtxt(
            f"{RAW_DATA_PATH}/{TEST_ID}/layer_{layer}/start_dir.csv",
            delimiter=','
            )
        layer_dict = {
                'dh':dh_data[layer][:],
                'vel_set':vel_set[layer,:],
                # 'vel_calc':vel_calc[layer,:],
                # 'avg_temp':temp[layer][1:-1],
                # 'max_temp':max_temp[layer][1:-1],
                'dir':direction
                }
        part_dict[f'l{layer}'] = layer_dict
    set_dict[PART_NAMES[idx]] = part_dict

savemat(f'datasets/aprbs_dataset.mat', set_dict)
