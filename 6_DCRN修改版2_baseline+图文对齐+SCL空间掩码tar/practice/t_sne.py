import time

import torch
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# %config InlineBackend.figure_format = "svg" # jupyter

def t_SNE(data, label):
    X_tsne = TSNE(perplexity=10, n_components=2, random_state=33).fit_transform(data)
    X_pca = PCA(n_components=2).fit_transform(data)

    font = {"color": "darkred",
            "size": 13,
            "family" : "serif"}

    # plt.style.use("dark_background")
    plt.style.use("default")
    plt.figure(figsize=(8.5, 4))


    plt.subplot(1, 2, 1)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=label, alpha=0.6,
                cmap=plt.cm.get_cmap('rainbow', 17)) # 17
    plt.title("t-SNE", fontdict=font)
    cbar = plt.colorbar(ticks=range(17)) # 17
    cbar.set_label(label='digit value', fontdict=font)
    plt.clim(-0.5, 16.5) # (-0.5, 15.5)


    plt.subplot(1, 2, 2)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=label, alpha=0.6,
                cmap=plt.cm.get_cmap('rainbow', 17)) # 17
    plt.title("PCA", fontdict=font)
    cbar = plt.colorbar(ticks=range(17)) # 17
    cbar.set_label(label='digit value', fontdict=font)
    plt.clim(-0.5, 16.5) # (-0.5, 15.5)
    plt.tight_layout()

    plt.show()

# # 如果在主程序导入这个，from practice import t_sne，会把里边的程序执行一般，路径就不对了。。。。。。
# data_embed_collect = []
# a = torch.randn(5000, 128)
# b = torch.randn(5000, 128)
# data_embed_collect.append(a)
# data_embed_collect.append(b)
# labels = np.random.randint(0,16, (10000)) # 正确，IP的标签就是0~15
#
# from utils import utils
# from sklearn.manifold import TSNE
#
# start_time = time.time()
#
# best_data_embed_collect_npy = torch.cat(data_embed_collect, axis = 0).cpu().numpy()
# n_samples, n_features = best_data_embed_collect_npy.shape
# # 调用t-SNE对高维的data进行降维，得到的2维的result_2D，shape=(samples,2)
# tsne_2D = TSNE(n_components=2, init='pca', random_state=0)
# result_2D = tsne_2D.fit_transform(best_data_embed_collect_npy)
# color_map = ['darkgray', 'lightcoral', 'salmon', 'peru', 'orange', 'gold', 'yellowgreen', 'darkseagreen',
#              'mediumaquamarine', 'skyblue', 'powderblue', 'thistle', 'plum', 'pink', 'darkgoldenrod', 'tomato']  # 16个类，准备16种颜色
# fig = utils.plot_embedding_2D(result_2D, labels, 'IP', color_map)
# fig.savefig("../tsne/SNE_random.png")
# fig.savefig("../tsne/SNE_random.pdf")
# # fig.show()
# end_time = time.time()
#
# print("OK")
# print("time consuming = ", end_time - start_time) # scatter time consuming =  147.67930960655212    plot time consuming = 33.393343448638916
#
