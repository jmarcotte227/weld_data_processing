import numpy as np
import matplotlib.pyplot as plt
from motoman_def import robot_obj, positioner_obj

if __name__=='__main__':
    DATASET = "2025_11_19_11_50_06_AL_WLJ_dataset0"
    REC_DIR = '../../../recorded_data/'
    LAYER = 2

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
    REC_DIR = '../'
    DATASET = ''
    joint_data = np.loadtxt(f"{REC_DIR}{DATASET}/weld_js.csv", delimiter=',', skiprows=1)

    t_start = joint_data[0,0]
    cart_vels = []
    cart_vels_cmd = []
    time_stamps=[]
    time_stamps_cmd=[]
    time_difs=[]
    time_difs_cmd=[]
    for idx in range(joint_data.shape[0]):
        robot1_pose=robot.fwd(joint_data[idx][1:7])
        time_stamp=joint_data[idx][0]
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
            # lin_vel = np.sqrt(cart_vel[0]**2+cart_vel[1]**2+cart_vel[2]**2)
            cart_vels.append(lin_vel)
            time_stamps.append(time_stamp)
            time_difs.append(time_dif)
    # for idx in range(joint_data_cmd.shape[0]):
    #     robot1_pose=robot.fwd(joint_data_cmd[idx][1:8])
    #     time_stamp=joint_data_cmd[idx][0]
    #     if idx==0:
    #         pose_prev=robot1_pose.p
    #         time_prev=time_stamp
    #     else:
    #         cart_dif=robot1_pose.p-pose_prev
    #         time_dif = time_stamp-time_prev
    #         time_prev = time_stamp
    #         cart_vel = cart_dif/time_dif
    #         time_prev = time_stamp
    #         pose_prev = robot1_pose.p
    #         lin_vel = np.sqrt(cart_vel[0]**2+cart_vel[1]**2)
    #         # lin_vel = np.sqrt(cart_vel[0]**2+cart_vel[1]**2+cart_vel[2]**2)
    #         cart_vels_cmd.append(lin_vel)
    #         time_stamps_cmd.append(time_stamp)
    #         time_difs_cmd.append(time_dif)

    plot_min = 0
    plot_max = -1
    
    fig, ax = plt.subplots(2,1, sharex=True)
    ax[0].scatter(time_stamps-t_start, cart_vels)
    # ax[0].scatter(time_stamps_cmd-t_start, cart_vels_cmd)
    # ax[0].scatter(time_stamps_cmd-t_start, v_cmd[1:])
    ax[1].scatter(time_stamps-t_start, time_difs)
    # ax[1].scatter(time_stamps_cmd-t_start, time_difs_cmd)
    ax[0].set_title("Velocitys")
    ax[0].set_ylabel("Velocity (mm/s)")
    ax[1].set_title("Time Step")
    ax[1].set_xlabel("Clock Time (s)")
    ax[1].set_ylabel("Time Step (s)")
    ax[0].legend(["Measured","Commanded Joint", "Commanded"])
    plt.show()


    # fig, ax = plt.subplots(1,1)
    # ax.hist(joint_data_cmd[1:,0]-joint_data_cmd[:-1,0], bins=100)
    # plt.show()
