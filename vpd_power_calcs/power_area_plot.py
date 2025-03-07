import numpy as np
import matplotlib.pyplot as plt

num_feedrate = 24
num_vel = 24
rad = 1.2/2 #mm

v_w = np.linspace(70, 300, num_feedrate)
I = np.array([32, 35, 38, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 73, 78, 83, 89, 98, 110, 122, 134, 142, 148, 153])
V = np.array([10.1, 10.3, 10.5, 10.7, 10.9, 11, 11.1, 11.1, 11.1, 11.2, 11.2, 11.2, 11.3, 11.6, 12.2, 12.8, 13.3, 13.8, 14.2, 14.5, 14.9, 15, 15.1, 15.2])
P = I*V

v_t = np.linspace(3,17, num_vel)

# calculate Q
Q = np.zeros((num_feedrate, num_vel))
for i in range(num_feedrate):
    for j in range(num_vel):
        Q[i,j] = P[i]/v_t[j]

# calcualte A
A = np.zeros((num_feedrate, num_vel))
for i in range(num_feedrate):
    for j in range(num_vel):
        A[i,j] = np.pi*(rad*rad)*(v_w[i]/2.362)/v_t[j]

# maintain constant v_w/v_t: crosssecitonal area
fig,ax = plt.subplots()
for j in [3,5, 10, 17]:
    init_S = 160/j
    v_t_set = []
    v_w_set = []
    Q_out = []

    for i in range(num_feedrate):
        v_t_i = v_w[i]/init_S
        if 3<v_t_i<17:
            print("S: ", v_w[i]/v_t_i)
            v_t_set.append(v_t_i)
            v_w_set.append(v_w[i])
            Q_out.append(P[i]/v_t_i)
    ax.plot(v_w_set, Q_out, label = f"S:{init_S:.2}+' mm^2, v_T=[{v_t_set[0]:.2},{v_t_set[-1]:.2}]")

    
ax.legend()
ax.set_title("Q at constant S, bounded to v_T in {3, 17}")
ax.set_xlabel("v_w (IPM)")
ax.set_ylabel("Q (J/mm)")

# fig, ax = plt.subplots()
# im = ax.imshow(np.log(Q), extent=(v_t[0], v_t[-1], v_w[-1], v_w[0]), aspect=0.05)
# cbar = ax.figure.colorbar(im, ax = ax, shrink=1)
# cbar.set_label("ln(J/mm)")
# ax.invert_yaxis()
# ax.set_xticks(v_t[::3])
# ax.set_yticks(v_w[::3])
# ax.set_xlabel("v_t(mm/s)")
# ax.set_ylabel("v_w(IPM)")
# ax.set_title("Energy Per Length")


# fig, ax = plt.subplots()
# im = ax.imshow(np.log(A), extent=(v_t[0], v_t[-1], v_w[-1], v_w[0]), aspect=0.05)
# cbar = ax.figure.colorbar(im, ax = ax, shrink=1)
# cbar.set_label("ln(A (mm^2))")
# ax.invert_yaxis()
# ax.set_xticks(v_t[::3])
# ax.set_yticks(v_w[::3])
# ax.set_xlabel("v_t(mm/s)")
# ax.set_ylabel("v_w(IPM)")
# ax.set_title("Cross-section Area")




plt.show()
