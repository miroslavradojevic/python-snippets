#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

# p = np.linspace(0.001, 0.999, num=100)
# print(type(p), p.shape)
# l = np.log(p/(1-p))
# plt.plot(p, l)
# plt.show()

N = 10
seq_on = np.ones(N) * .9
seq_off = np.ones(N) * .1
seq_init = np.ones(N) * .5

x = np.concatenate((seq_init, seq_off, seq_off, seq_on), axis=0)
y = np.zeros(x.shape)

for i in range(x.shape[0]):
    if i > 0:
        y[i] = 1. / (1. + (((1. - x[i]) / x[i]) * ((1. - x[i-1]) / y[i-1])))
    else:
        y[i] = 1. / (1. + (1. - x[i]) / x[i])

plt.plot(range(len(x)), y, linewidth=2)
plt.ylim(0.,1.)
plt.xticks(range(len(x)), x, rotation=90, fontsize=6)
plt.grid(linestyle='--', linewidth=.2)
plt.savefig("P_n.pdf", format='pdf', dpi=300, bbox_inches='tight')
plt.clf()

plt.plot(range(len(x)), x, color='green', linestyle='dashed', marker='o', markersize=4)
plt.ylim(0.,1.)

plt.grid(linestyle='--', linewidth=.2)
plt.xlabel('t')
plt.savefig("z_t.pdf", format='pdf', dpi=300, bbox_inches='tight')
plt.clf()