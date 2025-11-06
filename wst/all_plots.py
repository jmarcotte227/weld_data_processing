import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import cv2

from scipy import signal
from scipy.io import wavfile

# TEST_ID = 'weld_40IPM_404320250703-131217'
# TEST_ID = 'weld_100IPM_20250703-140141'
TEST_ID = 'KEEP_40IPM_braid_120250807-113051'
FRAMERATE = 90
EPS = 1e-1

### filepaths ###
data_dir = f'../../wst_data/{TEST_ID}/'

# load video
cap = cv2.VideoCapture(f'{data_dir}video_recording.avi')

# Read all frames 
video_frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_frames.append(frame)
cap.release()

vid_start = np.loadtxt(f'{data_dir}xiris_start_stop.csv', delimiter=',')[0]
vid_end = np.loadtxt(f'{data_dir}xiris_start_stop.csv', delimiter=',')[1]
vid_times = np.linspace(0, len(video_frames)/FRAMERATE, len(video_frames))

# load audio
sample_rate, samples = wavfile.read(f'{data_dir}mic_recording.wav')
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

# load spectra
spec_counts = np.loadtxt(f'{data_dir}spec_counts.csv', delimiter = ',', skiprows=1)
spec_wavelengths = np.loadtxt(f'{data_dir}spec_wavelengths.csv', delimiter = ',', skiprows=1)
wavelengths = spec_wavelengths[0,1:]
spec_times = spec_counts[:,0]-spec_counts[0,0]

other_start = spec_counts[0,0]
vid_times = vid_times-(other_start-vid_start)
# print(vid_times)

# setup plots
ax = []
gs = gridspec.GridSpec(2,2)
fig = plt.figure()
ax.append(plt.subplot(gs[:,0]))
ax.append(plt.subplot(gs[0,1]))
ax.append(plt.subplot(gs[1,1]))

# fig.subplots(sharex='col')
# plot spectrogram
ax[1].pcolormesh(times, frequencies, np.log(spectrogram))
line_1, = ax[1].plot([0,0], [frequencies[0], frequencies[-1]], alpha=0.5, color='r')
ax[1].set_title('Acoustic Spectrogram')
ax[1].set_ylabel('Frequency [Hz]')
ax[1].set_xlabel('Time [sec]')
ax[1].set_xlim(0,times[-1])
ax[2].pcolormesh(spec_times, wavelengths, spec_counts[:,1:].T)
line_2, = ax[2].plot([0,0], [wavelengths[0], wavelengths[-1]], alpha=0.5, color='r')
ax[2].set_title('Emission Spectrogram')
ax[2].set_ylabel('Wavelength [m]')
ax[2].set_xlabel('Time [sec]')
ax[1].set_xlim(0,times[-1])

# initialize video
video_im = ax[0].imshow(video_frames[0])
ax[0].axis('off')

# set animation function
def update(i):
    # update video frame
    if i<len(video_frames):
        video_im.set_data(video_frames[i])

    if vid_times[i]>0:
        line_1.set_data([vid_times[i], vid_times[i]], [frequencies[0], frequencies[-1]])
        line_2.set_data([vid_times[i], vid_times[i]], [wavelengths[0], wavelengths[-1]])

    return line_2,line_1,video_im,

# run animation
ani = animation.FuncAnimation(
    fig, update, frames=len(video_frames), interval=1/FRAMERATE*100, blit=True
)
fig.set_size_inches(10,5)
plt.tight_layout()
# plt.show()
# Set up formatting for the movie files
writer = animation.FFMpegWriter(fps=FRAMERATE)
ani.save(f'plots_{TEST_ID}.mp4', writer=writer)

plt.close()
