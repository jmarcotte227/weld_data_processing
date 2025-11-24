import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams['text.usetex'] = True

import sys
import yaml
import matplotlib.pyplot as plt
import numpy as np
from motoman_def import robot_obj, positioner_obj
from robotics_utils import H_inv
import scipy
import matplotlib.animation as animation
import pickle
from tqdm import tqdm

sys.path.append("../../../Welding_Motoman/toolbox/")
from angled_layers import avg_by_line

def main():
    REC_DIR = "../../../recorded_data/"
    # DATASET = "2025_11_19_11_50_06_AL_WLJ_dataset0"
    # DATASET = "2025_11_19_12_20_00_AL_WLJ_dataset1"
    # DATASET = "2025_11_19_12_50_06_AL_WLJ_dataset2"
    DATASET = "2025_11_19_13_19_53_AL_WLJ_dataset3"

    # load robot config
    CONFIG_DIR = f'{REC_DIR}{DATASET}/config/'
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

    vel_set = []

    for layer in tqdm(range(50)):
        cap_data=f"{REC_DIR}{DATASET}/layer_{layer}/"
        cart_vels = []
        job_no = []
        joint_angle = np.loadtxt(cap_data+'weld_js_exe.csv', delimiter=',')
        for idx in range(joint_angle.shape[0]):
            robot1_pose=robot.fwd(joint_angle[idx][3:9])
            time_stamp=joint_angle[idx][0]
            if idx==0:
                pose_prev=robot1_pose.p
                time_prev=time_stamp
            else:
                cart_dif=robot1_pose.p-pose_prev
                time_dif = time_stamp-time_prev
                time_prev = time_stamp
                cart_vel = cart_dif/time_dif
                time_prev = time_stamp
                pose_prev = robot1_pose.p
                lin_vel = np.sqrt(cart_vel[0]**2+cart_vel[1]**2)
                cart_vels.append(lin_vel)
                job_no.append(int(joint_angle[idx][1]))

        avg_vels = avg_by_line(np.array(job_no), np.array(cart_vels), np.linspace(0,49,50))
        vel_set.append(avg_vels)

    vel_set=np.squeeze(np.array(vel_set))

    np.savetxt(f"calc_v/{DATASET}_vel_calc.csv",vel_set, delimiter=',')
if __name__=="__main__":
    main()
