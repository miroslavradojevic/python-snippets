import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

reds = cm.get_cmap('Reds')
weight = np.linspace(0, 1, 10)
print(weight, "\n", reds(weight)[:, :3])