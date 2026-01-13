import sys, yaml
import numpy as np
import pickle
import glob
from motoman_def import robot_obj, positioner_obj
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append("../../../Welding_Motoman/toolbox/")
from angled_layers import avg_by_line

def main():
    TRIM = 0
    
    REC_DIR = "../../../recorded_data/"
    # DATASET = "2025_11_19_11_50_06_AL_WLJ_dataset0"
    # DATASET = "2025_11_19_12_20_00_AL_WLJ_dataset1"
    # DATASET = "2025_11_19_12_50_06_AL_WLJ_dataset2"
    # DATASET = "2025_11_19_13_19_53_AL_WLJ_dataset3"
    # DATASET = "2026_01_12_10_21_38_wall_lstm_control"
    DATASET = "2026_01_12_11_48_35_wall_lstm_baseline_control"

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
    flir_intrinsic = yaml.load(open(CONFIG_DIR + "FLIR_A320.yaml"), Loader=yaml.FullLoader)


    layer_dirs = glob.glob(f"{REC_DIR}{DATASET}/layer_*")
    num_layer = len(layer_dirs)

    # load flame data
    with open(f"proc_ir_vid/{DATASET}_flame.pkl", 'rb') as file:
        flame = pickle.load(file)

    fig,ax = plt.subplots()

    dhs = []
    hs = []
    nan_list=np.empty(46)
    nan_list[:]=np.nan
    dhs.append(nan_list)
    hs.append(nan_list)
    for layer in range(1,num_layer):
        prev_flame = flame[layer-1]
        # print("P: ", prev_flame)
        curr_flame = flame[layer]
        # print("C: ", curr_flame)

        prev_flame_avg = avg_by_line(prev_flame[:,0], prev_flame[:,1:], np.linspace(45,0,46)) 
        curr_flame_avg = avg_by_line(curr_flame[:,0], curr_flame[:,1:], np.linspace(0,45,46))

        # if layer%2:
        #     ax.plot(curr_flame_avg[:,2])
            # hs.append(curr_flame_avg[:,2])
        # else:
        #     ax.plot(np.flip(curr_flame_avg[:,2]))
            # hs.append(np.flip(curr_flame_avg[:,2]))
        ax.plot(curr_flame_avg[:,0], curr_flame_avg[:,2])
        dhs.append(curr_flame_avg[:,2]-prev_flame_avg[:,2])
        hs.append(curr_flame_avg[:,2])
    ax.set_xlabel("Segment Index")
    ax.set_ylabel("H")
    dhs=np.array(dhs)
    plt.show()
    np.savetxt(f"calc_dh/{DATASET}_dh.csv", dhs, delimiter=',')
    np.savetxt(f"calc_h/{DATASET}_h.csv", hs, delimiter=',')

    if True:
        fig,ax = plt.subplots(1,1)
        for layer in range(1,num_layer):
            # if layer%2:
            #     plt.plot(dhs[layer])
            # else:
            #     plt.plot(np.flip(dhs[layer]))
            ax.plot(dhs[layer])
        ax.set_xlabel("Segment Index")
        ax.set_ylabel("dH")
        ax.set_title("dH - Same Direction")
        plt.show()


if __name__=="__main__":
    main()
