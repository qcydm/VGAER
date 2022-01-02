from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
def community (z,clusters):
    z = z.detach().numpy()
    C_model = KMeans(n_clusters=clusters, verbose=1, max_iter=100, tol=0.01, n_init=3)
    C_model.fit(z)

    #train label
    commu_predict = C_model.labels_
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
    return commu_predict