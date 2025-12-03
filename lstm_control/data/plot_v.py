import numpy as np
import matplotlib.pyplot as plt

if __name__=='__main__':
    DATASET = "2025_11_19_11_50_06_AL_WLJ_dataset0"
    REC_DIR = '../../../recorded_data/'
    LAYER = 2

    calc_cmd_dataset = np.loadtxt(f"calc_v_cmd/{DATASET}_vel_calc_cmd.csv", delimiter=',')
    calc_dataset = np.loadtxt(f"calc_v/{DATASET}_vel_calc.csv", delimiter=',')
    set_dataset = np.loadtxt(f"v_set/{DATASET}_v_cmd.csv", delimiter=',')
    joint_data = np.loadtxt(f"{REC_DIR}{DATASET}/layer_{LAYER}/weld_js_exe.csv", delimiter=',')

    print(calc_dataset.shape)
    print(set_dataset.shape)
    v_calc = calc_dataset[LAYER, :]
    v_calc_cmd = calc_cmd_dataset[LAYER, :]
    v_set = set_dataset[LAYER, :]

    plot_min = 0
    plot_max = -1
    fig, ax = plt.subplots(1,1)
    ax.plot(v_calc[plot_min:plot_max])
    ax.plot(v_calc_cmd[plot_min:plot_max])
    ax.plot(v_set[plot_min:plot_max])
    ax.legend()
    plt.show()
    fig, ax = plt.subplots(1,1)
    ax.plot(joint_data[:,0], joint_data[:,3:9])
    plt.show()
    idx = np.where(joint_data[:,1]==20)
    disc_loc = joint_data[idx,3:9]

    # fig,ax = plt.subplots(1,1)
    # ax.plot(np.squeeze(joint_data[idx,0]), np.squeeze(disc_loc))
    # plt.show()


    fig, ax = plt.subplots(1,1)
    ax.hist(joint_data[1:,0]-joint_data[:-1,0], bins=20)
    plt.show()
    print(np.mean(joint_data[1:,0]-joint_data[:-1,0]))
