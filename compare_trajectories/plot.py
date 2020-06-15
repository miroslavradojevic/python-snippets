import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

DO_MIN_MAX_NORM = True

traj_1 = pd.read_csv('./_odometry_filtered_20200615-011217.csv')
traj_1.columns = ['timestamp','x', 'y', 'z']

traj_2 = pd.read_csv('./_orb_slam2_rgbd_pose_20200615-011214.csv')
traj_2.columns = ['timestamp','x', 'y', 'z']

if DO_MIN_MAX_NORM:
    scaler = MinMaxScaler()
    traj_1 = pd.DataFrame(scaler.fit_transform(traj_1), columns=traj_1.columns)
    traj_2 = pd.DataFrame(scaler.fit_transform(traj_2), columns=traj_2.columns)

fig = plt.figure(figsize=(12,8))
# ax = fig.gca()
# ax.set_xticks(np.arange(0, 1, 0.1))
# ax.set_yticks(np.arange(0, 1., 0.1))
plt.plot(traj_1['x'], traj_1['y'], label="RTAB-Map")
plt.plot(traj_2['x'], traj_2['y'], label="ORB2")
plt.legend(loc="upper left")
plt.grid()
plt.axis('equal')
plt.show()
fig.savefig("compare_xy.pdf", bbox_inches='tight')

fig1 = plt.figure(figsize=(12,8))
plt.plot(traj_1['timestamp'], traj_1['z'], label="RTAB-Map")
plt.plot(traj_2['timestamp'], traj_2['z'], label="ORB2")
plt.legend(loc="upper left")
plt.grid()
plt.axis('equal')
plt.show()
fig1.savefig("compare_z.pdf", bbox_inches='tight')





# sns.set(style="darkgrid")
# sns.scatterplot(x='x',y='y',data=traj_1)
# traj_1.info()
# traj_1.head()
# traj_2.info()
# traj_2.head()

# print(len(traj_1), type(traj_1))
# print(len(traj_2))
