import os
import numpy as np
import torch
import pickle as dill
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn import manifold
from collections import Counter

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# def plot_embedding_3d(X, title=None):
#     #坐标缩放到[0,1]区间
#     x_min, x_max = np.min(X,axis=0), np.max(X,axis=0)
#     X = (X - x_min) / (x_max - x_min)
#     #降维后的坐标为（X[i, 0], X[i, 1],X[i,2]），在该位置画出对应的digits
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1, projection='3d')
#     for i in range(X.shape[0]):
#         ax.text(X[i, 0], X[i, 1], X[i,2], str(digits.target[i]), color=plt.cm.Set1(y[i] / 10.), fontdict={'weight': 'bold', 'size': 9})
#     if title is not None:
#         plt.title(title)
#     plt.show()

if __name__ == '__main__':

    # T-SNE
    domains = ['D1', 'D2', 'D3', 'D4']

    data = []
    p_label = []
    d_label = []
    label = []

    for d in domains:
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
                if labels[i] == 'N' and c_N<c:
                    c_N += 1
                    data.append(samples[i])
                    p_label.append(p)
                    d_label.append(d)
                    label.append(labels[i])
                elif labels[i] == 'V' and c_V<c:
                    c_V += 1
                    data.append(samples[i])
                    p_label.append(p)
                    d_label.append(d)
                    label.append(labels[i])
                elif labels[i] == 'F' and c_F<c:
                    c_F += 1
                    data.append(samples[i])
                    p_label.append(p)
                    d_label.append(d)
                    label.append(labels[i])
                elif labels[i] == 'S' and c_S<c:
                    c_S += 1
                    data.append(samples[i])
                    p_label.append(p)
                    d_label.append(d)
                    label.append(labels[i])
                elif labels[i] == 'Q' and c_Q<c:
                    c_Q += 1
                    data.append(samples[i])
                    p_label.append(p)
                    d_label.append(d)
                    label.append(labels[i])
                else:
                    pass

    data = np.array(data)
    data = data.reshape([data.shape[0],data.shape[1]])
    tsne = manifold.TSNE(n_components=3, perplexity=30.0,
                 early_exaggeration=12.0, learning_rate=200.0, n_iter=1000,
                 n_iter_without_progress=300, min_grad_norm=1e-7,
                 metric="euclidean", init="random", verbose=0,
                 random_state=None, method='barnes_hut', angle=0.5,
                 n_jobs=None, square_distances='legacy')
    X_tsne = tsne.fit_transform(data)

    print("Org data dimension is {}.Embedded data dimension is {}".format(data.shape[-1], X_tsne.shape[-1]))

    '''嵌入空间可视化'''
    # colors = ['blue', 'cyan', 'green', 'red', 'yellow', 'magenta', 'black']
    # colors = ['limegreen', 'tomato', 'purple', 'yellow', 'orange', 'royalblue', 'black']
    colors = ['red', 'royalblue', 'limegreen', 'darkorange']

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    # plt.figure(figsize=(16, 16))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    for i in range(X_norm.shape[0]):
    # for i in range(100):
    #     if label[i] == 'N':
    #         plt.text(X_norm[i, 0], X_norm[i, 1], 'N', color=colors[domains.index(d_label[i])],
    #              fontdict={'weight': 'bold', 'size': 9}, label=d_label[i])
    #     elif label[i] == 'V':
    #         plt.text(X_norm[i, 0], X_norm[i, 1], 'V', color=colors[domains.index(d_label[i])],
    #                  fontdict={'weight': 'bold', 'size': 9}, label=d_label[i])
    #     elif label[i] == 'F':
    #         plt.text(X_norm[i, 0], X_norm[i, 1], 'F', color=colors[domains.index(d_label[i])],
    #                  fontdict={'weight': 'bold', 'size': 9}, label=d_label[i])
    #     elif label[i] == 'S':
    #         plt.text(X_norm[i, 0], X_norm[i, 1], 'S', color=colors[domains.index(d_label[i])],
    #                  fontdict={'weight': 'bold', 'size': 9}, label=d_label[i])
    #     elif label[i] == 'Q':
    #         plt.text(X_norm[i, 0], X_norm[i, 1], 'Q', color=colors[domains.index(d_label[i])],
    #                  fontdict={'weight': 'bold', 'size': 9}, label=d_label[i])

        ax.text(X_norm[i, 0], X_norm[i, 1], X_norm[i, 2], 'o', color=colors[domains.index(d_label[i])],
                 fontdict={'weight': 'bold', 'size': 2}, label=d_label[i])

    plt.xticks([])
    plt.yticks([])
    plt.legend(labels=domains)
    title = 'tsne_mit-bih'
    plt.title(title)
    plt.savefig(title + '.png')
    plt.show()
    plt.cla()







