import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.io import savemat

RAW_DATA_PATH = '../../recorded_data'
PROC_DATA_PATH = '../../Welding_Motoman/scan/angled_layer'
PART_NAMES = [
        'OL_cold',
        'OL_hot',
        'CL_cold',
        'CL_hot'
        ]
TEST_IDS = [
        'ER4043_bent_tube_large_cold_OL_2024_11_14_11_56_43',
        'ER4043_bent_tube_large_hot_OL_2024_11_14_13_05_38',
        'ER4043_bent_tube_large_cold_2024_11_07_10_21_39',
        'ER4043_bent_tube_large_hot_2024_11_06_12_27_19',
        ]

set_dict = {}
for idx, TEST_ID in enumerate(TEST_IDS):
    # load processed data (since it is compiled per part) 
    dh_data = np.loadtxt(f'{PROC_DATA_PATH}/calc_dh/{TEST_ID}_dhs.csv', delimiter=',')
    error_data = np.loadtxt(f'{PROC_DATA_PATH}/error_data/{TEST_ID}_layer_err.csv', delimiter=',')
    vel_calc = np.loadtxt(f'{PROC_DATA_PATH}/calc_vel/{TEST_ID}_vel_calc.csv', delimiter=',')
    temp = np.loadtxt(f'{PROC_DATA_PATH}/temp_data/{TEST_ID}_temps.csv', delimiter=',')
    max_temp = np.loadtxt(f'{PROC_DATA_PATH}/temp_data/{TEST_ID}_max_temps.csv', delimiter=',')
    voltage = np.loadtxt(f'{PROC_DATA_PATH}/weld_data_proc/{TEST_ID}_voltages.csv', delimiter=',')
    current = np.loadtxt(f'{PROC_DATA_PATH}/weld_data_proc/{TEST_ID}_currents.csv', delimiter=',')
    feedrate = np.loadtxt(f'{PROC_DATA_PATH}/weld_data_proc/{TEST_ID}_feedrates.csv', delimiter=',')
    energy = np.loadtxt(f'{PROC_DATA_PATH}/weld_data_proc/{TEST_ID}_energy.csv', delimiter=',')
    # with open(f'{PROC_DATA_PATH}temp_data/{TEST_ID}_temps.pkl', 'rb') as file:
    #     temp = pickle.load(file)

    layer_count = dh_data.shape[0]
    part_dict = {}

    for layer in range(1,layer_count):
        # load raw data (since it is compiled per layer)
        vel_set = np.loadtxt(f'{RAW_DATA_PATH}/{TEST_ID}/layer_{layer}/velocity_profile.csv', delimiter=',')
        # find layer direction
        direction = np.loadtxt(
            f"{RAW_DATA_PATH}/{TEST_ID}/layer_{layer}/start_dir.csv",
            delimiter=','
            )

        ### NOT FLIPPING DIRECTION FOR NOW, WILL HANDLE LATER IN LINE
        # # if the direction is opposite the normal direction
        # if not direction:
        #     layer_dict = {
        #             'dh':dh_data[layer],
        #             # 'error':error_data[layer],
        #             'vel_set':np.flip(vel_set[layer]),
        #             'vel_calc':np.flip(vel_calc[layer]),
        #             'temp':np.flip(temp[layer]),
        #             'dir':direction
        #             }
        # else:
        # error doesn't include the start and end, trimming from the others
        layer_dict = {
                'dh':dh_data[layer][1:-1],
                'error':error_data[layer],
                'vel_set':vel_set[1:-1],
                'vel_calc':vel_calc[layer][1:-1],
                'avg_temp':temp[layer][1:-1],
                'max_temp':max_temp[layer][1:-1],
                'voltage':voltage[layer][1:-1],
                'current':current[layer][1:-1],
                'feedrate':feedrate[layer][1:-1],
                'energy':energy[layer][1:-1],
                'dir':direction
                }
        part_dict[f'l{layer}'] = layer_dict
    set_dict[PART_NAMES[idx]] = part_dict

savemat(f'bent_tube_dataset.mat', set_dict)
