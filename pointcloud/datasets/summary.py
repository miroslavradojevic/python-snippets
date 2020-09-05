import pandas as pd
import matplotlib.pyplot as plt
import statistics as st
from os.path import splitext

fname = 'kitti05.csv'
fname = 'kitti07.csv'
fname = 'nsh.csv'
fname = 'LCAS_20160523_1200_1218.csv'
fname = 'LCAS_20160523_1227_1238.csv'
fname = 'LCAS_20160523_1239_1256.csv'
fname = 'InLiDa_sequence_1.csv'
fname = 'temposan_log_fri_02.csv'
fname = 'temposan_log_sat_02.csv'
dname = splitext(fname)[0]

df = pd.read_csv(fname)
df.columns = ['timestamp', 'points', 'points_dsamp']
fig = plt.figure(figsize=(8, 8))
plt.plot((df['timestamp']-df['timestamp'][0])/1e9, df['points'])
plt.title(dname)
plt.grid()
plt.xlabel("sec")
plt.ylabel("points")
fig.savefig(dname + ".png", bbox_inches='tight')

periods = df['timestamp'].diff()
periods = periods[1:] / 1e9 # seconds
freqs = 1. / periods

print("{}, med_p={}, mean_p={:.2f}, t_mean={:.3f}, {:.3f}Hz".format(dname, st.median(df['points']), st.mean(df['points']), st.mean(periods), st.mean(freqs)))
