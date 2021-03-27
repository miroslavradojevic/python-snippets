#!/usr/bin/env python
# generate random (x,y) path
import random
import math
import numpy as np
import matplotlib.pyplot as plt
N = 1000
step = 0.05
thetaSigma = np.pi/5

path = []
path.append([0, 0, 0]) # x, y, theta
# print(path, type(path), len(path))

def wrap(a):
    # https://stackoverflow.com/questions/58627711/get-angle-into-range-0-2pi-python
    a = a % (2*math.pi)  # wrap in [0, 2pi) range
    return a

for i in range(N):
    theta0 = path[-1][2]
    x0 = path[-1][0]
    y0 = path[-1][1]
    
    theta1 = wrap(theta0 + random.gauss(0, thetaSigma)) # random.uniform(0, 2 * np.pi)
    x1 = x0 + step * math.cos(theta1)
    y1 = y0 + step * math.sin(theta1)

    path.append([x1, y1, theta1])
    # print(np.asarray(path).shape)

xPath = np.asarray(path)[:,0]
yPath = np.array(path)[:,1]
# print(xPath.shape)
# print(np.asarray(path).shape)


fig = plt.figure(figsize=(8, 8))
plt.plot(xPath, yPath, marker='o', markersize=1)
plt.grid()
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.grid(linestyle='-.', linewidth=.2)
# plt.show()
fig.savefig("random_path.png", bbox_inches='tight')