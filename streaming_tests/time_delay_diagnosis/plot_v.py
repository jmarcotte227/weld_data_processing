import numpy as np
import matplotlib.pyplot as plt
from motoman_def import robot_obj, positioner_obj

if __name__=='__main__':
    # DATASET = "2026_01_23_10_46_41_WLJ_XX_motion_test_normal_priority_no_load"
    # DATASET = "2026_01_23_10_40_04_WLJ_XX_motion_test_high_priority_no_load"
    # DATASET = "2026_02_02_16_55_08_WLJ_XX_new_data_test_old_lam_cur"
    # DATASET = "2026_02_02_16_51_44_WLJ_XX_new_data_test_new_lam_cur"
    DATASET = "2026_02_17_15_15_48_WLJ_XX_new_data_test"
    # DATASET = "2026_01_28_14_38_23_WLJ_XX_new_data_test"
    # DATASET = "2026_01_23_10_26_27_WLJ_XX_motion_test_normal_priority_cpu_stressed"
    # DATASET = "2026_01_23_10_33_25_WLJ_XX_motion_test_high_priority_cpu_stressed"
    # DATASET = "2026_01_23_11_31_28_WLJ_XX_motion_test_normal_priority_linux"
    # DATASET = "2026_01_23_11_38_39_WLJ_XX_motion_test_high_priority_linux"
    # DATASET = "2025_11_19_11_50_06_AL_WLJ_dataset0"
    REC_DIR = '../../../recorded_data/'
    LAYER = 5
    CONFIG_DIR = '../../../adaptive_closed_loop_waam/config/'

    robot=robot_obj(
        'MA2010_A0',
        def_path=CONFIG_DIR+'MA2010_A0_robot_default_config.yml',
        tool_file_path=CONFIG_DIR+'torch.csv',
        pulse2deg_file_path=CONFIG_DIR+'MA2010_A0_pulse2deg_real.csv',
        d=10
    )
    # commanded joint data
    js_cmd = np.loadtxt(f"{REC_DIR}{DATASET}/layer_{LAYER}/weld_js_cmd.csv", delimiter=',')

    js_exe = np.loadtxt(f"{REC_DIR}{DATASET}/layer_{LAYER}/weld_js_exe.csv", delimiter=',')
    # js_exe = js_exe[::4]

    # end effector position
    cmd_pos = []
    exe_pos = []
    for idx, q in enumerate(js_cmd):
        cmd_pos.append(robot.fwd(q[2:8]).p)
    for idx, q in enumerate(js_exe):
        exe_pos.append(robot.fwd(q[3:9]).p)

    cmd_pos = np.array(cmd_pos)
    exe_pos = np.array(exe_pos)
    cmd_dts = js_cmd[1:,0]-js_cmd[:-1,0]
    exe_dts = js_exe[1:,0]-js_exe[:-1,0]
    cmd_vels = (cmd_pos[1:,0]-cmd_pos[:-1,0])/cmd_dts
    exe_vels = (exe_pos[1:,0]-exe_pos[:-1,0])/exe_dts
    plot_min = 0
    plot_max = -1

    fig, ax = plt.subplots(3,1, sharex=True)
    ax[0].plot(js_exe[:,0]-js_exe[0,0], exe_pos[:,0])
    ax[0].plot(js_cmd[:,0]-js_exe[0,0], cmd_pos[:,0])
    ax[0].plot(js_cmd[[0,-1],0]-js_exe[0,0], cmd_pos[[0,-1], 0], 'r--', alpha=0.5)
    # ax[0].plot(js_exe[:,0]-js_exe[0,0], js_exe[:,4])
    # ax[0].plot(js_cmd[:,0]-js_exe[0,0], js_cmd[:,3])
    # ax[0].plot(js_cmd[[0,-1],0]-js_exe[0,0], js_cmd[[0,-1], 3], 'r--', alpha=0.5)
    ax[1].plot(js_exe[1:,0]-js_exe[0,0],exe_vels)
    ax[1].plot(js_cmd[1:,0]-js_exe[0,0],cmd_vels)
    ax[1].plot(js_cmd[[1,-1],0]-js_exe[0,0], [10,10], 'r--', alpha=0.5)
    ax[2].plot(js_exe[1:,0]-js_exe[0,0], exe_dts)
    ax[2].plot(js_cmd[1:,0]-js_exe[0,0], cmd_dts)
    ax[2].plot(js_cmd[[1,-1],0]-js_exe[0,0], [0.008,0.008], 'r--', alpha=0.5)
    ax[2].plot(js_cmd[[1,-1],0]-js_exe[0,0], [0.004,0.004], 'g--', alpha=0.5)
    ax[2].set_xlabel("Time (s)")
    ax[0].set_ylabel("X Position (mm)")
    ax[1].set_ylabel("Torch Velocity (mm)")
    ax[2].set_ylabel("Sample Time (s)")
    fig.legend(["Measured", "Commanded"])
    fig.tight_layout()
    plt.show()
