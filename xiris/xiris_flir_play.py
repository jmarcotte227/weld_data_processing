import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation

SAVE_DIR = '../../recorded_data/2025_11_12_12_54_30_INWL_imaging_tests/layer_2/'

# flir data
with open(f'{SAVE_DIR}ir_recording.pickle', 'rb') as file:
    flir_recording = pickle.load(file)
flir_ts=np.loadtxt(f'{SAVE_DIR}/ir_stamps.csv', delimiter=',')

# xiris data
with open(f'{SAVE_DIR}ir_recording_2.pickle', 'rb') as file:
    xir_recording = pickle.load(file)
    xir_recording = [frame/10 for frame in xir_recording]

xir_ts=np.loadtxt(f'{SAVE_DIR}/ir_stamps_2.csv', delimiter=',')

# calculate framerate
xir_freq = 1/np.mean([xir_ts[1:]-xir_ts[:-1]])
print(f"Xiris Framerate: {xir_freq}")
print(f"Lapsed Time: {xir_ts[-1]-xir_ts[0]}")

num_frames = len(xir_ts)
cmap='cividis'

# generate min and max for each image and make color map
flir_min = np.min([frame.min() for frame in flir_recording])
flir_max = np.max([frame.max() for frame in flir_recording])
xir_min = 770#np.min([frame.min() for frame in xir_recording])
xir_max = 1400# np.max([frame.max() for frame in xir_recording])



fig,ax = plt.subplots(1,2, figsize=(15,5))

# get the first frame in each
im_flir = ax[0].imshow(flir_recording[0], cmap=cmap, vmin=flir_min, vmax=flir_max)
im_xir = ax[1].imshow(xir_recording[0], cmap=cmap, vmin=xir_min, vmax=xir_max)
ax[0].set_title("FLIR A320")
ax[1].set_title("XIR-1800")
ax[0].axis('off')
ax[1].axis('off')

cbar1 = fig.colorbar(im_flir, ax=ax[0], fraction=0.046, pad=0.04)
cbar1.set_label("Brightness")

cbar2 = fig.colorbar(im_xir, ax=ax[1], fraction=0.046, pad=0.04)
cbar2.set_label("Temperature (C)")
plt.tight_layout()

def init_animation():
    im_flir.set_data(flir_recording[0])
    im_xir.set_data(xir_recording[0])
    return im_flir, im_xir

def update_frames(frame_num):
    # flir_idx = frame_num
    # xir_idx = np.argmin(np.abs(xir_ts-flir_ts[flir_idx]))
    xir_idx = frame_num
    flir_idx = np.argmin(np.abs(flir_ts-xir_ts[xir_idx]))

    im_flir.set_data(flir_recording[flir_idx])
    im_xir.set_data(xir_recording[xir_idx])
    
    return im_flir, im_xir

ani = animation.FuncAnimation(
    fig,
    update_frames,
    init_func=init_animation,
    frames=num_frames,
    interval=1/xir_freq*1000,
    blit=True,
    repeat=False
)

# writer = animation.FFMpegWriter(fps=xir_freq, metadata=dict(artist='Matplotlib'), bitrate=1800)
# ani.save('xir_flir_cap.mp4', writer=writer)
plt.show()




