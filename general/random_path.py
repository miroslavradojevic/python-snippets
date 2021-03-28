#!/usr/bin/env python
### generate random (x,y) path
import random
import math
import numpy as np
import matplotlib.pyplot as plt
N = 1000
step = 0.05
thetaSigma = np.pi/5

path = []
path.append([0, 0, 0]) # x, y, theta

def wrap(a):
    # https://stackoverflow.com/questions/58627711/get-angle-into-range-0-2pi-python
    a = a % (2*math.pi)  # wrap in [0, 2pi) range
    return a

def generate_trajectory(x0, y0, theta0, Npoints, path_step, path_style):
    path = []
    path.append([x0, y0, theta0])
    
    for i in range(Npoints):
        if (path_style=="random_gauss"):
            theta1 = wrap(path[-1][2] + random.gauss(0, np.pi/4))
            x1 = path[-1][0] + path_step * math.cos(theta1)
            y1 = path[-1][1] + path_step * math.sin(theta1)
            path.append([x1, y1, theta1])
    
    out = np.asarray(path)[:,:2]

    print("points shape {}".format(out.shape))

    return out

p = generate_trajectory(0, 0, 0, N, step, "random_gauss")

for i in range(N):
    theta0 = path[-1][2]
    x0 = path[-1][0]
    y0 = path[-1][1]
    
    theta1 = wrap(theta0 + random.gauss(0, thetaSigma)) # random.uniform(0, 2 * np.pi)
    x1 = x0 + step * math.cos(theta1)
    y1 = y0 + step * math.sin(theta1)

    path.append([x1, y1, theta1])

xPath = np.asarray(path)[:,0]
yPath = np.array(path)[:,1]

fig = plt.figure(figsize=(8, 8))
plt.plot(xPath, yPath, marker='o', markersize=1)
plt.grid()
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.grid(linestyle='-.', linewidth=.2)
# plt.show()
fig.savefig("random_path.png", bbox_inches='tight')