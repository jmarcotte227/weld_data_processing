import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

RAW_DATA_PATH = '../recorded_data/'
PROC_DATA_PATH = '../Welding_Motoman/scan/angled_layer/'
PART_NAME = 'OL_cold'
TEST_ID = 'ER4043_bent_tube_large_hot_2024_11_06_12_27_19'

# load part 
dh_data = np.loadtxt(DATA_PATH+'layer_dh.csv', delimiter=',')
vel_set = np.loadtxt(RAW_DATA_PATH+'layer_vel.csv', delimiter=',')
vel_calc = np.loadtxt(PROC_DATA_PATH+'layer_vel_calc.csv', delimiter=',')
with open(f'{PROC_DATA_PATH}temp_data/{TEST_ID}_temps.pkl', 'rb') as file:
    temp = pickle.load(file)

layer_count = dh_data.shape[0]
part_dict = {}

for layer in range(layer_count):
    # find layer direction
    direction = np.loadtxt(
        "{RAW_DATA_PATH}{TEST_ID}/layer_{layer}/start_dir.csv",
        delimiter=','
        )

    # if the direction is opposite the normal direction
    if not direction:
        layer_dict = {
                'dh':dh_data[layer],
                'error':error_data[layer],
                'vel_set':np.flip(vel_set[layer]),
                'vel_calc':np.flip(vel_calc[layer]),
                'temp':np.flip(temp[layer]),
                'dir':direction
                }
    else:
        layer_dict = {
                'dh':dh_data[layer],
                'vel_set':vel_set[layer],
                'vel_calc':vel_calc[layer],
                'temp':temp[layer],
                'dir':direction
                }
    part_dict[f'l{layer}'] = layer_dict

savemat(f'{PART_NAME}.mat', part_dict)
