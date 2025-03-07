import matplotlib.pyplot as plt
import numpy as np
from motoman_def import robot_obj, positioner_obj

PLOT_TASK = True
PLOT_JOINT = False
DOWNSAMPLE_CONST = 2

## Load reference data
ref_data_dir = '../../../Welding_Motoman/data/bent_tube/slice_ER_4043_large_hot/curve_sliced_relative/'
test_curve = np.loadtxt(ref_data_dir+'slice2_0.csv', delimiter=',')

## Initialize Plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(test_curve[:,0],test_curve[:,1],test_curve[:,2],'b--')
markersize=5

CONFIG_DIR = '../../../Welding_Motoman/config/'
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

legends = ["Planned Path"]
time_diffs = []
## Load test data
recorded_data = f'../../../recorded_data/ER4043_bent_tube_large_hot_streaming_2025_03_04_14_33_32/layer_2/'
weld_js_exe = np.loadtxt(recorded_data+'weld_js_exe.csv', delimiter=',')
weld_js_cmd = np.loadtxt(recorded_data+'weld_js_cmd.csv', delimiter=',')

timestamps = weld_js_exe[::DOWNSAMPLE_CONST,1]-weld_js_exe[0,1] # time stamp in seconds (I think)
robot_js = weld_js_exe[::DOWNSAMPLE_CONST,2:8]
robot2_js = weld_js_exe[::DOWNSAMPLE_CONST,8:14]
positioner_js = weld_js_exe[::DOWNSAMPLE_CONST,14:16]

## Calculate planned cartesian path
rob1_pose = robot.fwd(weld_js_cmd[:,1:7], world=True)
rob2_pose = robot2.fwd(weld_js_cmd[:,7:13],world=True)
positioner_pose = positioner.fwd(weld_js_cmd[:,13:15], world=True)

data_points = len(positioner_pose.p_all)
planned_path = np.zeros((data_points,3))
for i in range(data_points):
    planned_path[i,:] = positioner_pose.R_all[i].T@(rob1_pose.p_all[i] - positioner_pose.p_all[i])

### Calculate Cartesian Path from Joint Space
rob1_pose = robot.fwd(robot_js, world=True)
rob2_pose = robot2.fwd(robot2_js,world=True)
positioner_pose = positioner.fwd(positioner_js, world=True)

data_points = len(positioner_pose.p_all)
torch_path = np.zeros((data_points,3))
for i in range(data_points):
    torch_path[i,:] = positioner_pose.R_all[i].T@(rob1_pose.p_all[i] - positioner_pose.p_all[i])

## Calculate Cartesian Velocity from Joint Space

js_vel = np.zeros((data_points-1,6))
cart_vel = np.zeros((data_points-1))
for i in range(data_points-1):
    js_vel[i,:] = (robot_js[i+1,:]-robot_js[i,:])/(timestamps[i+1]-timestamps[i])
    vel_point = robot.jacobian(robot_js[i,:])@js_vel[i,:]
    cart_vel[i]=np.sqrt(vel_point[3:].dot(vel_point[3:]))
    np.linalg.norm(vel_point[3:],ord=2)

## Calculate forward kinematics, then take difference in task space
# cart_vel = np.zeros((data_points-1))
# for i in range(data_points-1):
#     velocity = (rob1_pose.p_all[i]-rob1_pose.p_all[i+1])/(timestamps[i+1]-timestamps[i])
#     cart_vel[i]=np.sqrt(velocity[0:2].dot(velocity[0:2]))

## Compute Moving Average
window_width = 51
# print(f'{j} mm/s setpoint, average_velocity: {np.mean(cart_vel):.3f} mm/s')
cumsum_vec = np.cumsum(np.insert(cart_vel, 0, 0))
ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
idx_dif = (ma_vec.shape[0]-cart_vel.shape[0])/2

ax.plot(torch_path[:,0],torch_path[:,1],torch_path[:,2])

fig2, ax2 = plt.subplots(1,1)
ax2.scatter(timestamps[30:],cart_vel[29:], s=markersize)
avg_x = timestamps[int((window_width-1)/2+1):-int((window_width-1)/2)]
avg_y = ma_vec
print(avg_x.shape)
print(avg_y.shape)
ax2.scatter(avg_x[10:],avg_y[10:] , s=markersize)
# ax2.plot(timestamps[[1,-1]], [j,j],'r--')
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Torch Speed (mm/s)")
ax2.legend(["Raw Calculated Velocity", "Windowed Average", "Setpoint"])
# ax2.set_title(f"Setpoint: {j} mm/s")
if PLOT_TASK: fig2.show()

## Calculate Errors
e_sum = 0
for n in range(len(torch_path)):
    # if n == 0: print(np.linalg.norm(planned_path-torch_path[n,:],ord=2,axis=1))
    e_sum+=np.min(np.linalg.norm(planned_path-torch_path[n,:],ord=2,axis=1))
# print(f"{j} mm/s error: ", e_sum/len(torch_path))
# legends.append(f"{j} mm/s: e={e_sum/len(torch_path):.2f} mm")

# calculate measurement period
time_diffs.extend(timestamps[1:]-timestamps[:-1])
# if j==20:
#     fig_time, ax_time=plt.subplots(1,1)
#     ax_time.hist(timestamps[1:]-timestamps[:-1])
#     ax_time.set_xlabel("Sample time (s)")
#     ax_time.set_ylabel("Number of Samples")
#     print(f"Sample Time Average: {np.mean(timestamps[1:]-timestamps[:-1])}")
#     print(f"Sample Time Std Dev: {np.std(timestamps[1:]-timestamps[:-1])}")
#     print(f"Sample Time Min: {np.min(timestamps[1:]-timestamps[:-1])}")
#     print(f"Sample Time Max: {np.max(timestamps[1:]-timestamps[:-1])}")

#     # plot joint angles
#     fig_angles, ax_angles = plt.subplots(6,1, sharex=True)
#     for m in range(6):
#         ax_angles[m].scatter(timestamps[1:],js_vel[:,m], s=markersize)
#         ax_angles[m].set_title(f"Joint {m+1} Velocity")
#         ax_angles[m].set_ylabel("Velocity (rad/s)")
#     fig_angles.tight_layout()
plt.show()

ax.set_aspect('equal')
ax.legend(legends)

print("Measurement Frequency: ",1/np.mean(time_diffs))

if PLOT_TASK: plt.show()
