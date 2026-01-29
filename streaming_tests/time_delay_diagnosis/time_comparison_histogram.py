import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from motoman_def import robot_obj, positioner_obj
import time



if __name__=="__main__":

    DATASETS = [
        "2026_01_23_10_46_41_WLJ_XX_motion_test_normal_priority_no_load",
        "2026_01_23_10_40_04_WLJ_XX_motion_test_high_priority_no_load",
        "2026_01_23_10_26_27_WLJ_XX_motion_test_normal_priority_cpu_stressed",
        "2026_01_23_10_33_25_WLJ_XX_motion_test_high_priority_cpu_stressed",
        "2026_01_23_11_31_28_WLJ_XX_motion_test_normal_priority_linux",
        "2026_01_23_11_38_39_WLJ_XX_motion_test_high_priority_linux"
    ]
    plot_points = []
    fig, ax = plt.subplots(6,1, sharex=True)
    for idx, DATASET in enumerate(DATASETS):
        for layer in range(10):
            DATA_DIR = f"../../../recorded_data/{DATASET}/"
            J_IDX = 1
            CONFIG_DIR = '../../../Welding_Motoman/config/'
            SLICE_DATA = '../../../adaptive_closed_loop_waam/data/wall/1_5mm_slice/curve_sliced_relative/'


            test_slice = np.loadtxt(SLICE_DATA+'slice1_0.csv', delimiter=',')
            robot=robot_obj(
                'MA2010_A0',
                def_path=CONFIG_DIR+'MA2010_A0_robot_default_config.yml',
                tool_file_path=CONFIG_DIR+'torch.csv',
                pulse2deg_file_path=CONFIG_DIR+'MA2010_A0_pulse2deg_real.csv',
                d=15
            )

            js_exe= np.loadtxt(f"{DATA_DIR}layer_{layer}/weld_js_exe.csv", delimiter=',')
            js_cmd = np.loadtxt(f"{DATA_DIR}layer_{layer}/weld_js_cmd.csv", delimiter=',')
            ir2_stamps= np.loadtxt(f"{DATA_DIR}layer_{layer}/ir_stamps_2.csv", delimiter=',')
            # ir_stamps= np.loadtxt(f"{DATA_DIR}layer_{layer}/ir_stamps.csv", delimiter=',')

            if layer == 0:
                exe_stamps = js_exe[1:,0]-js_exe[:-1,0]
                cmd_stamps = js_cmd[1:,0]-js_cmd[:-1,0]
            else:
                exe_stamps = np.hstack([exe_stamps, js_exe[1:,0]-js_exe[:-1,0]])
                cmd_stamps = np.hstack([cmd_stamps, js_cmd[1:,0]-js_cmd[:-1,0]])
        stamps = cmd_stamps
        print("----------------------")
        print(DATASET)
        print("----------------------")
        print(f"Max:  {np.max(stamps)}")
        print(f"Min:  {np.min(stamps)}")
        print(f"Mean: {np.mean(stamps)}")
        print(f"Std.: {np.std(stamps)}")
        print("----------------------")

        plot_points.append(stamps)

        ax[idx].hist(stamps, bins=20, density=True)
    plt.show()

    fig,ax = plt.subplots(1,1)
    ax.plot([0.5,6.5], [0.008, 0.008], 'r--', alpha=0.1)
    ax.violinplot(plot_points, widths=1)
    ax.set_xticks(np.arange(6)+1)
    ax.set_xticklabels(
        [
            "Windows, Normal Priority, No Load",
            "Windows, High Priority, No Load",
            "Windows, Normal Priority, Full Load",
            "Windows, High Priority, Full Load",
            "Linux, Normal Priority, No Load",
            "Linux, High Priority, No Load",
        ],
        rotation=45,
        ha='right'
    )
    ax.set_ylabel("Sample Time (s)")
    plt.tight_layout()
    # ax.boxplot(plot_points)
    plt.show()


    # fig, ax = plt.subplots(1, 1)

    # min_time = np.min(np.hstack((exe_stamps, ir2_stamps, ir_stamps)))

    # for stamps in [exe_stamps, ir2_stamps, ir_stamps]:
    #     diffs = stamps[1:]-stamps[:-1]
    #     ax.plot(stamps[1:]-min_time, diffs)

    # commanded angles don't have time saved correctly
    # fig, ax = plt.subplots(3,1, sharex=True)
    # diffs = exe_stamps[1:]-exe_stamps[:-1]
    # ax[2].plot(exe_stamps[1:]-exe_stamps[0], diffs)
    # ax[1].plot(exe_stamps[1:]-exe_stamps[0], js_exe[1:,J_IDX+3])
    # ax[0].plot(exe_stamps[1:]-exe_stamps[0], (js_exe[1:,J_IDX+3]-js_exe[:-1,J_IDX+3])/diffs)

    # diffs = cmd_stamps[1:]-cmd_stamps[:-1]
    # ax[2].plot(cmd_stamps[1:]-cmd_stamps[0], diffs)
    # ax[1].plot(cmd_stamps[1:]-cmd_stamps[0], js_cmd[1:,J_IDX+2])
    # ax[0].plot(cmd_stamps[1:]-cmd_stamps[0], (js_cmd[1:,J_IDX+2]-js_cmd[:-1,J_IDX+2])/diffs)
    
    # # diffs = ir_stamps[1:]-ir_stamps[:-1]
    # # ax[2].plot(ir_stamps[1:]-ir_stamps[0], diffs)
    # # diffs = ir2_stamps[1:]-ir2_stamps[:-1]
    # # ax[2].plot(ir2_stamps[1:]-ir2_stamps[0], diffs)

    # ax[2].set_xlabel("Time (s)")
    # ax[2].set_ylabel("Sample Time (s)")
    # ax[1].set_ylabel("Joint 2 Position (rad)")
    # ax[0].set_ylabel("Joint 2 Velocity (rad/s)")
    # ax[2].legend(["exe", "cmd"])
    # fig.tight_layout()
    # plt.show()



        
