import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.io import savemat

RAW_DATA_PATH = '../../recorded_data'
PROC_DATA_PATH = '../../Welding_Motoman/scan/angled_layer'
PART_NAME = 'OL_cold'
TEST_ID = 'ER4043_bent_tube_large_hot_2024_11_06_12_27_19'

# load processed data (since it is compiled per part) 
dh_data = np.loadtxt(f'{PROC_DATA_PATH}/calc_dh/{TEST_ID}_dhs.csv', delimiter=',')
print(dh_data.shape)
error_data = np.loadtxt(f'{PROC_DATA_PATH}/error_data/{TEST_ID}_layer_err.csv', delimiter=',')
# vel_calc = np.loadtxt(PROC_DATA_PATH+'layer_vel_calc.csv', delimiter=',')
temp = np.loadtxt(f'{PROC_DATA_PATH}/temp_data/{TEST_ID}_temps.csv', delimiter=',')
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
            # 'vel_calc':vel_calc[layer],
            'avg_temp':temp[layer][1:-1],
            'dir':direction
            }
    part_dict[f'l{layer}'] = layer_dict

savemat(f'{PART_NAME}.mat', part_dict)
