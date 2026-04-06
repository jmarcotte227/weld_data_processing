import pickle
import matplotlib.pyplot as plt
import yaml
import sys
from tqdm import tqdm
import numpy as np
from motoman_def import robot_obj, positioner_obj
from robotics_utils import H_inv
sys.path.append('../../../Welding_Motoman/toolbox')
from angled_layers import avg_by_line, rotate, flame_tracking_stream, interpolate_heights

def rms_error(data):
    data = np.array(data)
    n = 0
    num = 0
    for i in data:
        if not np.isnan(i): 
            num = num + i**2
            n+=1
    return np.sqrt(num/n)

dataset = "bent_tube/"
sliced_alg = "slice_ER_4043_lstm/"
data_dir = "../../../Welding_Motoman/data/" + dataset + sliced_alg


# CONFIG_DIR = "../../../Welding_Motoman/config/"
# REC_DIR = "../../../recorded_data/2026_01_12_10_21_38_wall_lstm_control/"
# REC_DIR = "../../../recorded_data/wall_lstm_baseline_control_2025_11_05_12_38_13/"
# REC_DIR = "../../../recorded_data/wall_lstm_control_2025_11_05_13_17_59/"
# REC_DIR = "../../../recorded_data/wall_lstm_control_2025_10_31_14_30_40/"
# REC_DIR = "../../../recorded_data/wall_lstm_control_2025_10_31_13_34_50/"
# HEIGHT_OFFSET = -6.318382754974749 

# DATASET = "2026_02_12_16_06_04_tube_lstm_control"
DATASET = "2026_02_19_11_25_50_tube_lstm_control"
HEIGHT_OFFSET = -8.058484994710991
REC_DIR = f"../../../recorded_data/{DATASET}/"
CONFIG_DIR = f"{REC_DIR}/config/"
NUM_SEGS = 46

with open(data_dir + "sliced_meta.yml", "r") as file:
    slicing_meta = yaml.safe_load(file)
######## ROBOTS ########
# Define Kinematics
robot=robot_obj(
    'MA2010_A0',
    def_path=CONFIG_DIR+'MA2010_A0_robot_default_config.yml',
    tool_file_path=CONFIG_DIR+'torch.csv',
    pulse2deg_file_path=CONFIG_DIR+'MA2010_A0_pulse2deg_real.csv',
    d=15
)
robot2=robot_obj(
    'MA1440_A0',
    def_path=CONFIG_DIR+'MA1440_A0_robot_default_config.yml',
    tool_file_path=CONFIG_DIR+'flir.csv',
    pulse2deg_file_path=CONFIG_DIR+'MA1440_A0_pulse2deg_real.csv',
    base_transformation_file=CONFIG_DIR+'MA1440_pose.csv'
)
positioner=positioner_obj(
    'D500B',
    def_path=CONFIG_DIR+'D500B_robot_extended_config.yml',
    tool_file_path=CONFIG_DIR+'positioner_tcp.csv',
    pulse2deg_file_path=CONFIG_DIR+'D500B_pulse2deg_real.csv',
    base_transformation_file=CONFIG_DIR+'D500B_pose.csv'
)
flir_intrinsic = yaml.load(open(CONFIG_DIR + "FLIR_A320.yaml"), Loader=yaml.FullLoader)

H2010_1440 = H_inv(robot2.base_H)


# load recorded data
title=REC_DIR.removesuffix('/').removeprefix('../../../recorded_data/')

# slicing data
point_of_rotation = np.array(
        (slicing_meta["point_of_rotation"], slicing_meta["baselayer_resolution"]*slicing_meta["baselayer_num"])
    )
layer_angle = slicing_meta["layer_angle"]
base_thickness = slicing_meta["baselayer_resolution"]*slicing_meta["baselayer_num"]


height_errs = []
for layer in tqdm(range(100)):
    to_flat_angle = np.deg2rad(layer_angle*(layer+1))
    start_dir = np.loadtxt(f"{REC_DIR}layer_{layer}/start_dir.csv", delimiter=",")


    try:
        flame_3d_prev, _, job_no_prev = flame_tracking_stream(
                f"{REC_DIR}layer_{layer}/",
                robot,
                robot2,
                positioner,
                flir_intrinsic,
                HEIGHT_OFFSET
                )
        if flame_3d_prev.shape[0] == 0:
            raise ValueError("No flame detected")
    except ValueError as e:
        print(e)
        flame_3d_prev = None
        ir_error_flag = True
        height_err = np.zeros(slicing_meta["layer_length"])
    else:
        new_x, new_z = rotate(
            point_of_rotation, (flame_3d_prev[:, 1], flame_3d_prev[:, 2]), to_flat_angle
        )
        flame_3d_prev[:, 1] = new_x
        flame_3d_prev[:, 2] = new_z - base_thickness

        averages_prev = avg_by_line(job_no_prev, flame_3d_prev, np.linspace(0,NUM_SEGS-1,NUM_SEGS))
        heights_prev = averages_prev[:,2]
        if not start_dir: heights_prev = np.flip(heights_prev)
        
        # heights_prev = interpolate_heights(height_profile, heights_prev)
        # height error based on the build height of the previous layer
        height_err = -heights_prev
        height_errs.append(height_err)

np.savetxt(title+'_layer_err.csv',height_errs, delimiter=',')
