
#!/usr/bin/env python
# Examples:
# https://www.python-graph-gallery.com/circular-barplot-basic
# https://matplotlib.org/stable/gallery/pie_and_polar_charts/polar_bar.html#sphx-glr-gallery-pie-and-polar-charts-polar-bar-py
# https://stackoverflow.com/questions/30329673/how-to-set-the-axis-limit-in-a-matplotlib-plt-polar-plot
# plot properties:
# https://matplotlib.org/stable/api/projections_api.html
import numpy as np
import matplotlib.pyplot as plt

N = 12
bottom = 0.0
max_height = 1

theta = np.linspace(-np.pi, np.pi, N, endpoint=False)
radii = np.linspace(0, 2.0, N, endpoint=False) # max_height*np.random.rand(N)
width = (2*np.pi) / N

fig = plt.figure(figsize=(8,8))
ax = plt.subplot(111, polar=True)
ax.set_theta_zero_location("N") # ax.set_theta_offset(np.pi/2)
ax.set_theta_direction(-1) # counterclockwise
ax.set_thetalim(-np.pi, np.pi)
ax.set_thetagrids((-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180))
ax.set_yticks(np.linspace(0, 1.0 , 3, endpoint=True))
ax.set_ylim(0,1.2)
bars = ax.bar(theta, radii, width=width, bottom=bottom)

# Use custom colors and opacity
# for r, bar in zip(radii, bars):
#     bar.set_facecolor(plt.cm.jet(r / 10.))
#     bar.set_alpha(0.8)

# plt.show()
fig.savefig("circular.png", bbox_inches='tight')