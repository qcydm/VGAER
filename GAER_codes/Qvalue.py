import numpy as np

# def load_csv(path):
#     data_read = pd.read_csv(path,header=None)
#     list = data_read.values.tolist()
#     data = np.array(list)
#     print(data.shape)
#     #print(data)
#     return data
# A=load_csv('adj.csv')
# Member=load_csv('membership.csv')

def Q(array, cluster):

    # 总边数
    m = sum(sum(array)) / 2
    k1 = np.sum(array, axis=1)
    k2 = k1.reshape(k1.shape[0], 1)
    # 节点度数积
    k1k2 = k1 * k2
    # 任意两点连接边数的期望值
    Eij = k1k2 / (2 * m)
    # 节点v和w的实际边数与随机网络下边数期望之差
    B = array - Eij
    # 获取节点、社区矩阵
    node_cluster = np.dot(cluster, np.transpose(cluster))
    results = np.dot(B, node_cluster)
    # 求和
    sum_results = np.trace(results)
    # 模块度计算
    Q = sum_results / (2 * m)
    print("Q:", Q)
    return Q


# if __name__ == '__main__':
#     # 邻接矩阵，2表示节点2和节点3之间有两条边相连
#     array = A
#
#     # 节点类别分别是1,2,2
#     cluster = Member
#     Q(array, cluster)