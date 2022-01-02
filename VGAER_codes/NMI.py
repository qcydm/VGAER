import math
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from sklearn import metrics
import matplotlib.pyplot as plt
from collections import Counter


def load_label(dataset):
    names = ['y', 'ty', 'ally']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset,names[i]), 'rb') as rf:
            u = pkl._Unpickler(rf)
            u.encoding = 'latin1'
            cur_label = u.load()
            objects.append(cur_label)
    y, ty, ally = tuple(objects)
    label_hot = sp.vstack((ty, ally))
    label_onehot = np.vstack((ty,ally))
    labels = []
    for i in range(label_hot.shape[0]):
        labels.append(label_hot.col[i])
    labels = np.array(labels)

    return labels, label_onehot




def NMI(A,B):
    A = np.array(A)
    B = np.array(B)
    #样本点数
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    #互信息计算
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A==idA)
            idBOccur = np.where(B==idB)
            idABOccur = np.intersect1d(idAOccur,idBOccur)
            px = 1.0*len(idAOccur[0])/total
            py = 1.0*len(idBOccur[0])/total
            pxy = 1.0*len(idABOccur)/total
            MI = MI + pxy*math.log(pxy/(px*py)+eps,2)
    # 标准化互信息
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0*len(np.where(A==idA)[0])
        Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps,2)
    Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0*len(np.where(B==idB)[0])
        Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps,2)
    MIhat = 2.0*MI/(Hx+Hy)
    print(MIhat)

    # plt.figure(figsize=(8,3), dpi=120)
    #
    # #（1）orgin
    # plt.subplot(1, 2, 1)  # 画布分为1行，2列，共2格，当前绘图区设定为第1格
    # plt.scatter(z[:, 0], z[:, 1], c="blue", marker='o', s=10)  # 形状是圆圈；圆圈大小是10；颜色是蓝色
    # plt.title("cora")
    # plt.subplot(1, 2, 2)  # 当前绘图区设定为第2格
    # plt.scatter(z[:, 0], z[:, 1], c=commu_predict, marker='o', s=10)  # 不同类别不同颜色
    # plt.title("k-means")


    return MIhat

def label_change(pred, obje):
    clusters = 7
    sort_pre = Counter(pred).most_common(clusters)
    sort_obj = Counter(obje).most_common(clusters)
    change_dict = {}
    for i in range(clusters):
        change_dict[sort_pre[i][0]] = sort_obj[i][0]
    new_pred = [change_dict[i] if change_dict else i for i in pred]

    return new_pred





#print(NMI(A,B))
