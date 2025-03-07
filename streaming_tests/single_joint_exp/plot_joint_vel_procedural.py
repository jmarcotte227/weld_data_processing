import os
import numpy as np
import matplotlib.pyplot as plt

data_dir='../../../recorded_data/streaming/streaming_delay/'
# data_dir='../../../recorded_data/streaming/'
STREAMING_RATE = 125.

for name in os.listdir(data_dir):
    try:
        s=name.split('_')
        joint=int(s[2])
        vel=s[4]
        cmd_data = np.loadtxt(f'{data_dir}{name}/weld_js_cmd.csv',delimiter=',')
        cmd_stamps = cmd_data[:,0]
        cmd_joints = cmd_data[:,joint+1]
        
        exe_data = np.loadtxt(f'{data_dir}{name}/weld_js_exe.csv',delimiter=',')
        exe_stamps = exe_data[:,0]
        exe_joints = exe_data[:,joint+2]

        # calculate time difference
        exe_time_diff = exe_stamps[1:]-exe_stamps[:-1]
        cmd_time_diff = cmd_stamps[1:]-cmd_stamps[:-1]

        # artificial time difference from STREAMING_RATE
        # time_exe_stream = np.linspace(0, (len(cmd_stamps)-1)/STREAMING_RATE, len(cmd_stamps))
        # time_diff_stream = time_exe_stream[1:]-time_exe_stream[:-1] 
        # print(time_diff_stream[:5])
        print(cmd_time_diff[:5])
    
        # calculate joint difference
        exe_joint_diff = exe_joints[1:]-exe_joints[:-1]
        cmd_joint_diff = cmd_joints[1:]-cmd_joints[:-1]

        # calculate velocity
        exe_vel = exe_joint_diff/exe_time_diff
        cmd_vel = cmd_joint_diff/cmd_time_diff
        # cmd_vel_adjust = cmd_joint_diff/time_diff_stream


        
        print(exe_stamps.dtype)
        print(exe_joints.dtype)
        fig,ax=plt.subplots()
        ax.scatter(exe_stamps[1:]-exe_stamps[0],exe_vel,s=1)
        ax.scatter(cmd_stamps[1:]-cmd_stamps[0],cmd_vel,s=1)
        # ax.scatter(time_exe_stream[1:],cmd_vel_adjust,s=1)
        ax.set_xlabel('time (s)')
        # ax.set_ylim([0,0.02])
        ax.set_ylabel('velocity (rad/s)')
        ax.set_title(f'Joint {joint+1} | Velocity {vel} rad/s')
        ax.legend(['Executed', 'Commanded'])
        fig.savefig(f'procedural_plots/velocity_plots/velocity_plots_corrected/joint_{joint+1}_vel_{vel}.png')
        # fig.savefig(f'procedural_plots/velocity_plots/joint_{joint+1}_vel_{vel}.png')
        # plt.show()
        # exit()
    except FileNotFoundError:
        pass
    except IndexError:
        pass
