import pickle
import matplotlib.pyplot as plt
import yaml
import sys
import numpy as np
from motoman_def import robot_obj, positioner_obj
from robotics_utils import H_inv
sys.path.append('../../Welding_Motoman/toolbox')
from angled_layers import avg_by_line, rotate, LiveAverageFilter

def rms_error(data):
    data = np.array(data)
    n = 0
    num = 0
    for i in data:
        if not np.isnan(i):
            num = num + i**2
            n+=1
    return np.sqrt(num/n)

config_dir = "../../Welding_Motoman/config/"
dataset = "bent_tube/"
sliced_alg = "slice_ER_4043_large_hot/"
data_dir = "../../Welding_Motoman/data/" + dataset + sliced_alg

# flame_set = 'processing_data/ER4043_bent_tube_2024_09_04_12_23_40_flame.pkl'
flame_set = [
    #'../processing_data/ER4043_bent_tube_2024_08_28_12_24_30_flame.pkl',
    # '../processing_data/ER4043_bent_tube_2024_09_04_12_23_40_flame.pkl',
    # 'processing_data/ER4043_bent_tube_2024_09_03_13_26_16_flame.pkl',
    # '../processing_data/ER4043_bent_tube_hot_2024_10_21_13_25_58_flame.pkl'
    # 'processing_data/ER4043_bent_tube_large_hot_2024_11_06_12_27_19_flame.pkl',
    # '../processing_data/ER4043_bent_tube_large_cold_2024_11_07_10_21_39_flame.pkl'
    # '../processing_data/ER4043_bent_tube_large_cold_OL_2024_11_14_11_56_43_flame.pkl'
    # '../processing_data/ER4043_bent_tube_large_hot_OL_2024_11_14_13_05_38_flame.pkl'
    'processing_data/ER4043_bent_tube_large_hot_streaming_2025_03_06_feedback_troubleshooting_flame.pkl'
]
title=flame_set[-1].removesuffix('_flame.pkl').removeprefix('../processing_data/')
with open(data_dir + "slicing.yml", "r") as file:
    slicing_meta = yaml.safe_load(file)

robot = robot_obj(
    "MA2010_A0",
    def_path=config_dir+"MA2010_A0_robot_default_config.yml",
    tool_file_path=config_dir+"torch.csv",
    pulse2deg_file_path=config_dir+"MA2010_A0_pulse2deg_real.csv",
    d=15,
)
robot2 = robot_obj(
    "MA1440_A0",
    def_path=config_dir+"MA1440_A0_robot_default_config.yml",
    tool_file_path=config_dir+"flir_imaging.csv",
    pulse2deg_file_path=config_dir+"MA1440_A0_pulse2deg_real.csv",
    base_transformation_file=config_dir+"MA1440_pose.csv",
)
positioner = positioner_obj(
    "D500B",
    def_path=config_dir+"D500B_robot_default_config.yml",
    tool_file_path=config_dir+"positioner_tcp.csv",
    pulse2deg_file_path=config_dir+"D500B_pulse2deg_real.csv",
    base_transformation_file=config_dir+"D500B_pose.csv",
)

H2010_1440 = H_inv(robot2.base_H)
H = np.loadtxt(data_dir + "curve_pose.csv", delimiter=",")
p = H[:3, -1]
R = H[:3, :3]

layer_start = 1
rms_errs = []
# fig,ax = plt.subplots()

for idx,flame in enumerate(flame_set):
    with open(flame, 'rb') as file:
        flames = pickle.load(file)
    print("Flames Loaded, plotting")
    # print(len(flames[43]))
    # exit()

    # Rotation parameters
    job_no_offset = 0
    point_of_rotation = np.array(
            (slicing_meta["point_of_rotation"], slicing_meta["baselayer_thickness"]))
    base_thickness = slicing_meta["baselayer_thickness"]
    layer_angle = np.array((slicing_meta["layer_angle"]))
    print(layer_angle)

    curve_sliced = np.loadtxt(data_dir+"curve_sliced/slice1_0.csv", delimiter=',')
    dist_to_por = []
    for i in range(len(curve_sliced)):
        point = np.array((curve_sliced[i, 0], curve_sliced[i, 2]))
        dist = np.linalg.norm(point - point_of_rotation)
        dist_to_por.append(dist)

    # height_profile = []
    # for distance in dist_to_por:
    #     height_profile.append(distance * np.sin(np.deg2rad(layer_angle)))
    height_err = []
    height_err_trim = []
    flames_flat = []
    flames_filter = []
    # for layer, flame in enumerate(flames):
    for layer, flame in enumerate(flames):
        # plt.plot(flame[:,1], flame[:,3])
        to_flat_angle = np.deg2rad(layer_angle*(layer+1))
        for i in range(flame.shape[0]):
            flame[i,1:] = R.T @ flame[i,1:]

        new_x, new_z = rotate(
            point_of_rotation, (flame[:, 1], flame[:, 3]), to_flat_angle
        )
        # plt.plot(new_x, new_z)
        flame[:, 1] = new_x
        flame[:, 3] = new_z - base_thickness
        flame[:,0] = flame[:,0]-job_no_offset
        if layer == 103:
            filter = LiveAverageFilter()
            flame_filt_val = [0,0,0]
            curr_idx = 0
            seg_flame_count = 0

            for i in range(flame.shape[0]):
                if (flame[i,0]!=curr_idx):
                    flame_filt_val = filter.read_filter()
                    for j in range(seg_flame_count+1):
                        flames_filter.append(flame_filt_val)
                    curr_idx = flame[i,0]
                    seg_flame_count=0
                else:
                    seg_flame_count += 1
                    filter.log_reading(flame[i,1:])
            flame_filt_val = filter.read_filter()
            for j in range(seg_flame_count):
                flames_filter.append(flame_filt_val)

            flames_filter = np.array(flames_filter)
            plt.plot(flame[:,0],flame[:,3])
            plt.plot(flame[:,0],flames_filter[:,2])
            plt.xlabel("Timestep")
            plt.ylabel("Error (mm)")
            plt.title(f"Filter Comparison Layer {layer}")
            plt.show()
        # plt.plot(flame[:,1], flame[:,3])
        averages= avg_by_line(flame[:,0], flame[:,1:], np.linspace(0,49,50))
        height_err.append(averages[:,2])
        flames_flat.append(averages)
        # if layer == 75:
        #     plt.plot(averages[:,0], averages[:,2])
        #     plt.show()
    rms_err = []
    for scan in height_err:
        rms_err.append(rms_error(scan[1:-1]))
        height_err_trim.append(scan[1:-1])
    rms_errs.append(rms_err)
# ax.set_aspect('equal')
# plt.show()

fig,ax = plt.subplots()

# np.savetxt(title+'_err.csv',rms_errs[0])
# np.savetxt(title+'_layer_err.csv',height_err_trim, delimiter=',')
