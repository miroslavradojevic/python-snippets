
#!/usr/bin/env python
# Examples:
# https://www.python-graph-gallery.com/circular-barplot-basic
# https://matplotlib.org/stable/gallery/pie_and_polar_charts/polar_bar.html#sphx-glr-gallery-pie-and-polar-charts-polar-bar-py

import numpy as np
import matplotlib.pyplot as plt

N = 12
bottom = 0.0
max_height = 1

theta = np.linspace(-np.pi, np.pi, N, endpoint=False)
radii = max_height*np.random.rand(N)
width = (2*np.pi) / N

fig = plt.figure(figsize=(8,8))
ax = plt.subplot(111, polar=True)
bars = ax.bar(theta, radii, width=width, bottom=bottom)

# Use custom colors and opacity
# for r, bar in zip(radii, bars):
#     bar.set_facecolor(plt.cm.jet(r / 10.))
#     bar.set_alpha(0.8)

# plt.show()
fig.savefig("circular.png", bbox_inches='tight')