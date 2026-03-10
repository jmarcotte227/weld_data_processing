import numpy as np
import cv2
import pickle 
from glob import glob
from motoman_def import robot_obj

def main():
    # DATASET = "2025_06_11_16_27_WL0SS_datacollection"
    # DATASET = "2025_06_11_17_16_WL0SS_datacollection"
    DATASET = "2026_02_26_12_05_19_weld_wall_ER4043"

    LAYER = 14  
    SCALE_FACT = 2

    BASE_DIR = f"../../recorded_data/{DATASET}/weld_data/baselayer_1/proc_data/"
    DATA_DIR = glob(f"../../recorded_data/{DATASET}/weld_data/layer_{LAYER}_*/")[0]
    print(DATA_DIR)
    CONFIG_DIR = f'../../adaptive_closed_loop_waam/config/'


    # find average height over middle section of wall
    base_height = np.loadtxt(f"{BASE_DIR}/profile_height.csv", delimiter=',')
    start_idx, end_idx = find_start_end_idx(base_height[:,0], -50, 50)
    base_height = base_height[start_idx:end_idx, 1]

    layer_height = np.loadtxt(f"{DATA_DIR}/proc_data/profile_height.csv", delimiter=',')
    start_idx, end_idx = find_start_end_idx(layer_height[:,0], -50, 50)
    layer_height = layer_height[start_idx:end_idx, 1]

    print(f"Average Height: {np.mean(layer_height)-np.mean(base_height)}")

    # load the IR video
    with open(f"{DATA_DIR}/raw_data/ir_recording.pickle", 'rb') as file:
        ir_vid = pickle.load(file)

    # min max scaling on IR video
    ir_vid = (ir_vid-np.min(ir_vid))/(np.max(ir_vid)-np.min(ir_vid))

    ir_stamps = np.loadtxt(f"{DATA_DIR}/raw_data/ir_stamps.csv", delimiter=',')

    # load jointspace and robot config
    js_exe = np.loadtxt(f"{DATA_DIR}/raw_data/weld_js_exe.csv", delimiter=',')

    robot=robot_obj(
        'MA2010_A0',
        def_path=CONFIG_DIR+'MA2010_A0_robot_default_config.yml',
        tool_file_path=CONFIG_DIR+'torch.csv',
        pulse2deg_file_path=CONFIG_DIR+'MA2010_A0_pulse2deg_real.csv',
        d=10
    )

    init_pos = robot.fwd(js_exe[0,1:7]).p
    print(init_pos)

    window = "Viewer"
    cv2.namedWindow(window)
    frame = cv2.resize(
        ir_vid[0],
        np.flip([x*SCALE_FACT for x in ir_vid[0].shape]),
        interpolation=cv2.INTER_LINEAR
    )
    frame = cv2.putText(
        frame,
        f"Pos.: {np.linalg.norm(init_pos-robot.fwd(js_exe[0,1:7]).p)}",
        (10,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255,255,255),
        1
    )
    cv2.imshow(window, frame)
    trackbar_name = "Time"

    def trackbar_func(val):
        stamp = ir_stamps[val]
        j_idx = np.argmin(np.abs(js_exe[:,0]-ir_stamps[val]))

        frame = cv2.resize(
            ir_vid[val],
            np.flip([x*SCALE_FACT for x in ir_vid[val].shape]),
            interpolation=cv2.INTER_LINEAR
        )
        frame = cv2.putText(
            frame,
            f"Pos.: {np.linalg.norm(init_pos-robot.fwd(js_exe[j_idx,1:7]).p)}",
            (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255,255,255),
            1
        )
        cv2.imshow(window, frame)

    cv2.createTrackbar(trackbar_name, window, 0, len(ir_stamps)-1, trackbar_func)
    cv2.waitKey()

def find_start_end_idx(_list, start_pos, end_pos):
    start_idx = np.argmin(np.abs(_list-start_pos))
    end_idx = np.argmin(np.abs(_list-end_pos))
    return (start_idx, end_idx)


if __name__=="__main__":
    main()
