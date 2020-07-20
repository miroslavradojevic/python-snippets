import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# recordings_dir = "/Users/miroslav/source/python-snippets/compare_trajectories/nsh_indoor_outdoor.bag"
# recordings_dir = "/Users/miroslav/source/python-snippets/compare_trajectories/nsh_indoor_outdoor.bag/aloam"
recordings_dir = "/Users/miroslav/source/python-snippets/compare_trajectories/InLiDa-sequence_4"
# recordings_dir = "/Users/miroslav/source/python-snippets/compare_trajectories/temposan-log_fri_02.bag"

# list csv files in recordings_dir
csv_files = [f for f in os.listdir(recordings_dir) if f.endswith('.csv')]
print(csv_files)
#
trajectories = dict()
for f in csv_files:
    trajectories[os.path.splitext(f)[0]] = pd.read_csv(os.path.join(recordings_dir, f))
    trajectories[os.path.splitext(f)[0]].columns = ['timestamp', 'x', 'y', 'z']

markers = ['+', '.', 'o', '*', 'x', 'd', '1'] # ',',

fig = plt.figure(figsize=(8,8))
for key in trajectories:
    # print(markers[np.randint(len(markers))])
    plt.plot(trajectories[key]['x'], trajectories[key]['y'], marker=markers[np.random.randint(len(markers))], label=key,markevery=50)

plt.legend(loc="upper left")
plt.grid()
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.show()
fig.savefig(os.path.join(recordings_dir, "compare" + ".pdf"), bbox_inches='tight') # + str(DO_MIN_MAX_NORM)
