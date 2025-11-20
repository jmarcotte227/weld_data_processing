import sys, yaml
import numpy as np
import pickle
import glob
from motoman_def import robot_obj, positioner_obj
from tqdm import tqdm


sys.path.append("../../../Welding_Motoman/toolbox/")
from angled_layers import flame_tracking_stream

def main():
    REC_DIR = "../../../recorded_data/"
    # DATASET = "2025_11_19_11_50_06_AL_WLJ_dataset0"
    # DATASET = "2025_11_19_12_20_00_AL_WLJ_dataset1"
    # DATASET = "2025_11_19_12_50_06_AL_WLJ_dataset2"
    DATASET = "2025_11_19_13_19_53_AL_WLJ_dataset3"
    HEIGHT_OFFSET = -7.92870911432761

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
    
    flames = []

    for layer in tqdm(range(num_layer)):
        save_path = f"{REC_DIR}{DATASET}/layer_{layer}/"
        try:
            flame_3d, _, job_no = flame_tracking_stream(
                save_path,
                robot,
                robot2,
                positioner,
                flir_intrinsic,
                HEIGHT_OFFSET
            )
            if flame_3d.shape[0]==0:
                raise ValueError("no flame detected")
        except ValueError as e:
            print(e)
            flame_3d=None
        else:
            job_no = job_no.reshape((-1,1))
            output_array = np.hstack((job_no, flame_3d))
            flames.append(output_array)
    with open(f"proc_ir_vid/{DATASET}_flame.pkl", 'wb') as file:
        pickle.dump(flames, file)

if __name__=="__main__":
    main()


