import argparse
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import exists, isdir, splitext, join

# recordings_dir = "/Users/miroslav/source/python-snippets/compare-trajectories/nsh_indoor_outdoor.bag"
# recordings_dir = "/Users/miroslav/source/python-snippets/compare-trajectories/nsh_indoor_outdoor.bag/aloam"
# recordings_dir = "/Users/miroslav/source/python-snippets/compare-trajectories/temposan-log_fri_02.bag"
# recordings_dir = "/compare-trajectories/InLiDa-sequence_4"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_dir", help="Path to csv file", type=str)
    args = parser.parse_args()

    if not exists(args.csv_dir) or not isdir(args.csv_dir):
        exit(args.csv_dir + " could not be found")

    # list csv files
    csv_files = [f for f in os.listdir(args.csv_dir) if f.endswith('.csv')]
    print(csv_files)

    trajectories = dict()
    for f in csv_files:
        trajectories[splitext(f)[0]] = pd.read_csv(join(args.csv_dir, f))
        trajectories[splitext(f)[0]].columns = ['timestamp', 'x', 'y', 'z']

    markers = ['+', '.', 'o', '*', 'x', 'd', '1']  # ',',
    # TODO set random colors for future
    cols = {'UKF': '#EE6C4D', 'RTABMAP': '#293241', 'FLOAM': '#98C1D9', "KITTI_GT": "#FF0000"}

    # # rotate traj_x to align
    import math
    if True:
        alpha = (18.0 / 360.0) * math.pi * 2
        trajectories['UKF']['x1'] = trajectories['UKF']['x'] * math.cos(alpha) - trajectories['UKF']['y'] * math.sin(alpha)
        trajectories['UKF']['y1'] = trajectories['UKF']['y'] * math.cos(alpha) + trajectories['UKF']['x'] * math.sin(alpha)

        trajectories['UKF']['x1'] += 5
        trajectories['UKF']['y1'] += 5

        trajectories['UKF']['x'] = trajectories['UKF']['x1']
        trajectories['UKF']['y'] = trajectories['UKF']['y1']

    fig = plt.figure(figsize=(8, 8))
    for key in trajectories:
        # print(markers[np.randint(len(markers))])
        plt.plot(trajectories[key]['x'], trajectories[key]['y'],
                 # marker=markers[np.random.randint(len(markers))],
                 c=cols[key],
                 label=key)  # , markevery=50

    # key = 'UKF'
    # plt.plot(trajectories[key]['x1'], trajectories[key]['y1'],
    #          # marker=markers[np.random.randint(len(markers))],
    #          c=cols[key],
    #          label=key)

    plt.legend(loc="upper left")
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.show()
    fig.savefig(os.path.join(args.csv_dir, "compare" + ".png"), bbox_inches='tight')  # + str(DO_MIN_MAX_NORM)
