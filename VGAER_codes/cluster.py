from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
#def community (z,clusters):
    #z = z.detach().numpy()
    #C_model = KMeans(n_clusters=clusters, verbose=1, max_iter=100, tol=0.01, n_init=3)
    #C_model.fit(z)

    #train label
    #commu_predict = C_model.labels_
    # centers = C_model.cluster_centers_
    # # centers_id = []
    # # for i in range(clusters):
    # #     centers_id.append(int(np.argwhere(z==centers[i])[0]))
    # distance = C_model.inertia_
    # iterations = C_model.n_iter_
    # print("ceners = ", centers)
    # print("distance = ", distance)
    # print("iterations = ", iterations)

    # plt.figure(figsize=(8,3), dpi=120)
    #
    # #（1）orgin
    # plt.subplot(1, 2, 1)  # 画布分为1行，2列，共2格，当前绘图区设定为第1格
    # plt.scatter(z[:, 0], z[:, 1], c="blue", marker='o', s=10)  # 形状是圆圈；圆圈大小是10；颜色是蓝色
    # plt.title("cora")
    # plt.subplot(1, 2, 2)  # 当前绘图区设定为第2格
    # plt.scatter(z[:, 0], z[:, 1], c=commu_predict, marker='o', s=10)  # 不同类别不同颜色
    # plt.title("k-means")
    #return commu_predict
def community (z1, clusters):
    z1 = z1.detach().numpy()
    #for i in range(1,11):
    #ts = manifold.TSNE(n_components=2, perplexity=3, early_exaggeration=10, n_iter=50000, learning_rate=500,  angle=0.5, init='random')
    ts = TSNE(n_components=2, perplexity=50, early_exaggeration=500, n_iter=100000, learning_rate=100, angle=0.5,
                       init='random')
    z = ts.fit_transform(z1)
    #C_model = KMeans(n_clusters=clusters, verbose=0, max_iter=1000, tol=0.001, n_init=20, init='k-means++')
    C_model = KMeans(n_clusters=clusters, verbose=0, max_iter=1000, tol=0.001, n_init=20, init='k-means++')
    #C_model = KMeans(n_clusters=clusters, verbose=0, max_iter=100, tol=0.01, n_init=3)
    C_model.fit(z)
    commu_predict = C_model.labels_
    #torch.save(commu_predict, './pred.pt')
    plt.figure(figsize=(10,10), dpi=80)
    z = plt.scatter(z[:, 0], z[:, 1], c=commu_predict, marker='o', s=10)  # 不同类别不同颜色
    plt.title("k-means")
    #plt.savefig('./cora{}.pdf'.format(i))
    plt.show()
    print(i)
    return commu_predict
