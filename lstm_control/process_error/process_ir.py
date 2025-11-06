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

dataset = "wall/"
sliced_alg = "1_5mm_slice/"
data_dir = "../../../Welding_Motoman/data/" + dataset + sliced_alg

HEIGHT_OFFSET = -7.92870911432761 

CONFIG_DIR = "../../../Welding_Motoman/config/"
REC_DIR = "../../../recorded_data/wall_lstm_control_2025_10_31_14_30_40/"
# REC_DIR = "../../../recorded_data/wall_lstm_control_2025_10_31_13_34_50/"

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

height_errs = []
for layer in tqdm(range(slicing_meta["layer_num"])):
    start_dir = np.loadtxt(f"{REC_DIR}layer_{layer}/start_dir.csv", delimiter=",")


    build_height = (layer+1)*slicing_meta["layer_resolution"]\
        +slicing_meta["baselayer_num"]*slicing_meta["baselayer_resolution"]
    height_profile = np.ones(slicing_meta["layer_length"])*build_height

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
        averages_prev = avg_by_line(job_no_prev, flame_3d_prev, np.linspace(0,slicing_meta['layer_length']-1,slicing_meta['layer_length']))
        heights_prev = averages_prev[:,2]
        if not start_dir: heights_prev = np.flip(heights_prev)


        heights_prev = interpolate_heights(height_profile, heights_prev)
        height_err = np.ones(len(heights_prev))*build_height-heights_prev


        height_errs.append(height_err)

np.savetxt(title+'_layer_err.csv',height_errs, delimiter=',')
