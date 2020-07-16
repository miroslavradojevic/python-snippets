import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

recordings_dir = "/home/miro/stack/nuctech/progress/200713/loam-compare/kitti-2011_09_30_0027.bag"

DO_MIN_MAX_NORM = False

traj_1 = pd.read_csv(os.path.join(recordings_dir, 'FLOAM.csv'))
traj_1.columns = ['timestamp','x', 'y', 'z']

traj_2 = pd.read_csv(os.path.join(recordings_dir, 'ALOAM.csv'))
traj_2.columns = ['timestamp','x', 'y', 'z']

traj_3 = pd.read_csv(os.path.join(recordings_dir, 'KITTI_GT.csv'))
traj_3.columns = ['timestamp','x', 'y', 'z']


# rotate traj_3 - to align
# import math
# alpha = (-114.0 / 360.0) * math.pi * 2

# traj_3['x1'] = traj_3['x'] * math.cos(alpha) - traj_3['y'] * math.sin(alpha)
# traj_3['y1'] = traj_3['y'] * math.cos(alpha) + traj_3['x'] * math.sin(alpha)


if DO_MIN_MAX_NORM:
    scaler = MinMaxScaler()
    traj_1 = pd.DataFrame(scaler.fit_transform(traj_1), columns=traj_1.columns)
    traj_2 = pd.DataFrame(scaler.fit_transform(traj_2), columns=traj_2.columns)
    traj_3 = pd.DataFrame(scaler.fit_transform(traj_3), columns=traj_3.columns)

fig = plt.figure(figsize=(8,8))
plt.plot(traj_1['x'], traj_1['y'], label="FLOAM")
plt.plot(traj_2['x'], traj_2['y'], label="ALOAM")
plt.plot(traj_3['x'], traj_3['y'], label="KITTI_GT")
plt.legend(loc="upper left")
plt.grid()
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.show()
fig.savefig(os.path.join(recordings_dir, "compare_xy_minmax_" + str(DO_MIN_MAX_NORM) + ".pdf"), bbox_inches='tight')

fig1 = plt.figure(figsize=(8,8))
plt.plot(traj_1['timestamp'], traj_1['z'], label="FLOAM")
plt.plot(traj_2['timestamp'], traj_2['z'], label="ALOAM")
plt.plot(traj_3['timestamp'], traj_3['z'], label="KITTI_GT")
plt.legend(loc="upper left")
plt.grid()
plt.xlabel("timestamp")
plt.ylabel("z")
plt.axis('equal')
plt.show()
fig1.savefig(os.path.join(recordings_dir, "compare_z_minmax_" + str(DO_MIN_MAX_NORM) + ".pdf"), bbox_inches='tight')