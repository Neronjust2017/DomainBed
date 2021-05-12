from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import numpy as np
import os
import pandas as pd
from scipy import stats, integrate
import seaborn as sns
import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号

a = np.zeros([100, 2])
b = np.array([False, True])

c = a[:, b]

print(2)

# s1 = np.array([0., 0, 1, 2, 1, 0, 1, 0, 0, 2, 1, 0, 0])
# s2 = np.array([0., 1, 2, 3, 1, 0, 0, 0, 2, 1, 0, 0, 0])
# path = dtw.warping_path(s1, s2)
# dtwvis.plot_warping(s1, s2, path, filename="warp.png")

# s1 = [0, 0, 1, 2, 1, 0, 1, 0, 0]
# # s2 = [0, 1, 2, 0, 0, 0, 0, 0, 0]
# s2 = [0, 0, 1, 1, 1, 0, 1, 0, 0]
# distance = dtw.distance(s1, s2)
# print(distance)

# from dtaidistance import dtw
# import numpy as np
# series = [
#     np.array([0, 0, 1, 2, 1, 0, 1, 0, 0], dtype=np.double),
#     np.array([0.0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0]),
#     np.array([0.0, 0, 1, 2, 1, 0, 0, 0])]
# ds = dtw.distance_matrix_fast(series)
# print(ds)

domains = ['D1', 'D2', 'D3', 'D4']

data = [[[] for j in range(5)] for i in range(len(domains))]

for i_d, d in enumerate(domains):
    patient = os.listdir(d)

    for p in patient:
        samples = np.load(os.path.join(d, p, 'samples.npy'))
        labels = np.load(os.path.join(d, p, 'labels.npy'))
        labels = list(labels)

        c_N = 0
        c_V = 0
        c_F = 0
        c_S = 0
        c_Q = 0
        c = 100

        for i in range(len(samples)):
            if labels[i] == 'N' and c_N < c:
                c_N += 1
                data[i_d][0].append(samples[i])
            elif labels[i] == 'V' and c_V < c:
                c_V += 1
                data[i_d][1].append(samples[i])
            elif labels[i] == 'F' and c_F < c:
                c_F += 1
                data[i_d][2].append(samples[i])
            elif labels[i] == 'S' and c_S < c:
                c_S += 1
                data[i_d][3].append(samples[i])
            elif labels[i] == 'Q' and c_Q < c:
                c_Q += 1
                data[i_d][4].append(samples[i])
            else:
                pass

# domain class sample
# s1 = np.array(data[0][3][0])
# s2 = np.array(data[3][3][0])
#
# s1 = s1.reshape([s1.shape[0], ])
# s2 = s2.reshape([s2.shape[0], ])

# path = dtw.warping_path(s1, s2)
# dtwvis.plot_warping(s1, s2, path, filename="d1-d4_S.png")
# ds = dtw.distance(s1, s2)

print(2)

# D1 N
# ts = np.array(data[0][0])
# ts = ts.reshape([ts.shape[0], ts.shape[1]])
# ds_matrix_d1 = dtw.distance_matrix_fast(ts)
# ds_d1 = []
# for i in range(ds_matrix_d1.shape[0]):
#     for j in range(ds_matrix_d1.shape[0]):
#         if i<j:
#             ds_d1.append(ds_matrix_d1[i][j])
#
# print(ds_matrix_d1.shape)
# print(ds_d1)

# D1-D2 N
# ts_d1 = np.array(data[0][0])
# ts_d1 = ts_d1.reshape([ts_d1.shape[0], ts_d1.shape[1]])
# 
# ts_d2 = np.array(data[1][0])
# ts_d2 = ts_d2.reshape([ts_d2.shape[0], ts_d2.shape[1]])
# 
# ts = np.concatenate([ts_d1, ts_d2], axis=0)
# 
# ds_matrix = dtw.distance_matrix_fast(ts)

# # N
# ts_d1 = np.array(data[0][0])
# ts_d1 = ts_d1.reshape([ts_d1.shape[0], ts_d1.shape[1]])
#
# ts_d2 = np.array(data[1][0])
# ts_d2 = ts_d2.reshape([ts_d2.shape[0], ts_d2.shape[1]])
#
# ts_d3 = np.array(data[2][0])
# ts_d3 = ts_d3.reshape([ts_d3.shape[0], ts_d3.shape[1]])
#
# ts_d4 = np.array(data[3][0])
# ts_d4 = ts_d4.reshape([ts_d4.shape[0], ts_d4.shape[1]])
#
# num_d1 = ts_d1.shape[0]
# num_d2 = ts_d2.shape[0]
# num_d3 = ts_d3.shape[0]
# num_d4 = ts_d4.shape[0]
#
# ts = np.concatenate([ts_d1, ts_d2, ts_d3, ts_d4], axis=0)
# ds = dtw.distance_matrix_fast(ts)
#
# ds_d1 = ds[:num_d1, :num_d1]
# ds_d1_d2 = ds[:num_d1, num_d1:num_d1+num_d2]
#
# ds_d1 = ds_d1.flatten()
# ds_d1_d2 = ds_d1_d2.flatten()
#
# sns.distplot(ds_d1, hist=False, rug=True, color="g")
# sns.distplot(ds_d1_d2, hist=False, rug=True, color="m")
#
# plt.show()

# # V
# ts_d1 = np.array(data[0][1])
# ts_d1 = ts_d1.reshape([ts_d1.shape[0], ts_d1.shape[1]])
#
# ts_d2 = np.array(data[1][1])
# ts_d2 = ts_d2.reshape([ts_d2.shape[0], ts_d2.shape[1]])
#
# ts_d3 = np.array(data[2][1])
# ts_d3 = ts_d3.reshape([ts_d3.shape[0], ts_d3.shape[1]])
#
# ts_d4 = np.array(data[3][1])
# ts_d4 = ts_d4.reshape([ts_d4.shape[0], ts_d4.shape[1]])
#
# num_d1 = ts_d1.shape[0]
# num_d2 = ts_d2.shape[0]
# num_d3 = ts_d3.shape[0]
# num_d4 = ts_d4.shape[0]
#
# ts = np.concatenate([ts_d1, ts_d2, ts_d3, ts_d4], axis=0)
# ds = dtw.distance_matrix_fast(ts)
#
# ds_d1 = ds[:num_d1, :num_d1]
# ds_d1_d2 = ds[:num_d1, num_d1:num_d1+num_d2]
#
# ds_d1 = ds_d1.flatten()
# ds_d1_d2 = ds_d1_d2.flatten()
#
# sns.distplot(ds_d1, hist=False, rug=True, color="g")
# sns.distplot(ds_d1_d2, hist=False, rug=True, color="m")
#
# plt.show()

# F
ts_d1 = np.array(data[0][3])
ts_d1 = ts_d1.reshape([ts_d1.shape[0], ts_d1.shape[1]])

ts_d2 = np.array(data[1][3])
ts_d2 = ts_d2.reshape([ts_d2.shape[0], ts_d2.shape[1]])

ts_d3 = np.array(data[2][3])
ts_d3 = ts_d3.reshape([ts_d3.shape[0], ts_d3.shape[1]])

ts_d4 = np.array(data[3][3])
ts_d4 = ts_d4.reshape([ts_d4.shape[0], ts_d4.shape[1]])

num_d1 = ts_d1.shape[0]
num_d2 = ts_d2.shape[0]
num_d3 = ts_d3.shape[0]
num_d4 = ts_d4.shape[0]

ts = np.concatenate([ts_d1, ts_d2, ts_d3, ts_d4], axis=0)
ds = dtw.distance_matrix_fast(ts)

ds_d1 = ds[:num_d1, :num_d1]
ds_d1_d2 = ds[:num_d1, num_d1:num_d1+num_d2]

ds_d1 = ds_d1.flatten()
ds_d1_d2 = ds_d1_d2.flatten()

sns.distplot(ds_d1, hist=False, rug=True, color="g")
sns.distplot(ds_d1_d2, hist=False, rug=True, color="m")

plt.show()