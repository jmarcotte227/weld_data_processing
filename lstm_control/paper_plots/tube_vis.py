import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns

def main():
    TUBE_D = 46 # mm

    print(TUBE_D)

    DATA_DIR = "data/"

    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    # rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)

    # colors = ["#7fc97f","#beaed4","#fdc086"]
    colors = ["#66c2a5","#fc8d62","#8da0cb"]

    # dataset
    test_data_1 = torch.load(f"{DATA_DIR}tube_Log-Log_Baseline_noise_True_pmodel_h-8_part-0_loss-0.2000_cmodel_h-8_part-1_loss-0.068420260107-232739/test_results.pt")
    test_data_2 = torch.load(f"{DATA_DIR}tube_Linearized_QP_Control_noise_True_pmodel_h-8_part-0_loss-0.2000_cmodel_h-8_part-1_loss-0.068420260107-232814/test_results.pt")

    datasets = [test_data_1, test_data_2]
    
    fig, ax = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
    for idx, d_set in enumerate(datasets):
        h = d_set["H"].numpy()
        # print(type(test_data["H_d"][0]))
        # print((test_data["H_d"][0]))

        # min max dh
        dh_min = np.min(d_set["H_d"][0].numpy(), axis=0)
        dh_max = np.max(d_set["H_d"][0].numpy(), axis=0)

        # angle and point of rotation
        layer_ang = np.arctan((dh_max-dh_min)/TUBE_D)
        rot_point = dh_max/np.tan(layer_ang)-TUBE_D/2
        

        # size of dataset
        layer_len = h.shape[1]
        num_layers = h.shape[0]
        print(f"Final layer height: {np.rad2deg(layer_ang*num_layers)}")

        dep_points = np.zeros((num_layers, layer_len, 3))

        # calculate dh
        h_prev = np.zeros((num_layers, layer_len))
        h_prev[1:, :] = h[:-1,:]
        dh = h-h_prev

        # calculate circle points
        flat_coords = circ_map(layer_len, TUBE_D/2)

        # fig_flat, ax_flat = plt.subplots(1,1)

        for layer in range(num_layers):
            # h is the distance up the tube, x and y determined by the segment
            x,z = point_from_height(h[layer], flat_coords, rot_point)

            dep_points[layer, :, 0] = x
            dep_points[layer, :, 1] = flat_coords[:,1]
            dep_points[layer, :, 2] = z

            # if layer%10==0: ax[idx].plot(x, flat_coords[:,1], z)
            ax[idx].plot(x, flat_coords[:,1], z)
            # ax_flat.plot(h[layer])

    for a in ax:
        a.set_aspect("equal")
        a.set_xlabel("X")
        a.set_ylabel("Y")
        a.set_zlabel("Z")
        a.view_init(elev=10., azim=-30)
    ax[0].set_title("Log-Log Baseline")
    ax[1].set_title("LSTM MPC")
    plt.tight_layout()
    plt.savefig(f"output_plots/tube_vis.png", dpi=300)
    plt.show()


def circ_map(num_segs, tube_r):
    # calculate x and y coordinates of each point along a circle
    # assumes num_segs are the complete circle starting and ending in
    #   almost the same spot. One off from the beginning
    rad_per_point = 2*np.pi/num_segs
    coords = np.zeros((num_segs, 2))
    for seg_idx in range(num_segs):
        # x coordinate
        coords[seg_idx,0] = tube_r*np.cos(rad_per_point/2+rad_per_point*seg_idx)
        coords[seg_idx,1] = tube_r*np.sin(rad_per_point/2+rad_per_point*seg_idx)

    return coords
    

def point_from_height(h, circ_points, rot_point):
    x = np.zeros_like(h)
    z = np.zeros_like(h)
    for idx, h_val in enumerate(h):
        rad = rot_point-circ_points[idx,0]
        ang = h_val/rad # angle in radians

        # rotate x,0 up that many degrees to get the point on the tube

        x_coord, z_coord = rotate(
                [rot_point, 0],
                (circ_points[idx, 0], 0),
                -ang
        )

        x[idx] = x_coord
        z[idx] = z_coord
    return x,z

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy

if __name__=="__main__":
    main()
