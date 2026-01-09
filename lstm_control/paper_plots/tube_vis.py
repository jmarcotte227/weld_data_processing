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
    test_data = torch.load(f"{DATA_DIR}tube_Log-Log_Baseline_noise_True_pmodel_h-8_part-0_loss-0.2000_cmodel_h-8_part-1_loss-0.068420260105-143325/test_results.pt")
    # test_data = torch.load(f"{DATA_DIR}tube_Linearized_QP_Control_noise_True_pmodel_h-8_part-0_loss-0.2000_cmodel_h-8_part-1_loss-0.068420260105-133232/test_results.pt")
    # test_data = torch.load(f"{DATA_DIR}tube_Linearized_QP_Control_noise_True_pmodel_h-8_part-0_loss-0.2000_cmodel_h-8_part-1_loss-0.068420260105-132924/test_results.pt")
    # test_data = torch.load(f"{DATA_DIR}tube_Linearized_QP_Control_noise_True_pmodel_h-8_part-0_loss-0.2000_cmodel_h-8_part-1_loss-0.068420260105-110930/test_results.pt")



    h = test_data["H"].numpy()

    # min max dh
    dh_min = np.min(test_data["H_d"][0].numpy(), axis=0)
    dh_max = np.max(test_data["H_d"][0].numpy(), axis=0)

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

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    fig_flat, ax_flat = plt.subplots(1,1)

    for layer in range(num_layers):
        # h is the distance up the tube, x and y determined by the segment
        x,z = point_from_height(h[layer], flat_coords, rot_point)

        dep_points[layer, :, 0] = x
        dep_points[layer, :, 1] = flat_coords[:,1]
        dep_points[layer, :, 2] = z

        ax.plot(x, flat_coords[:,1], z)
        ax_flat.plot(h[layer])

    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
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
