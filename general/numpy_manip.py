#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

N = 12
theta = np.linspace(-np.pi, np.pi, N, endpoint=False)

tau = -np.cos(theta)

fig = plt.figure(figsize=(8, 6))
plt.plot(theta, tau, c='b', marker='o', ls='--')
plt.grid()
plt.xlabel("theta")
plt.ylabel("tau")
plt.axis('equal')
fig.savefig("tau.png", bbox_inches='tight')

rho = tau
rho[(theta>=0) & (theta<np.pi)] = -tau[(theta>=0) & (theta<np.pi)]

fig = plt.figure(figsize=(8, 6))
plt.plot(theta, rho, c='b', marker='o', ls='--')
plt.grid()
plt.xlabel("theta")
plt.ylabel("rho")
plt.axis('equal')
fig.savefig("rho.png", bbox_inches='tight')
