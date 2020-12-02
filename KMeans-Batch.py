import torch
import numpy as np

def euclidean_metric_np(X, centroids):
    X = np.expand_dims(X, 1)
    centroids = np.expand_dims(centroids, 0)
    dists = (X - centroids) ** 2
    dists = np.sum(dists, axis=2)
    return dists


def euclidean_metric_gpu(X, centers):
    X = X.unsqueeze(1)
    centers = centers.unsqueeze(0)

    dist = torch.sum((X - centers) ** 2, dim=-1)
    return dist


def kmeans_fun_gpu(X, K=10, max_iter=1000, batch_size=8096, tol=1e-40):
    N = X.shape[0]

    indices = torch.randperm(N)[:K]
    init_centers = X[indices]

    batchs = N // batch_size
    last = 1 if N % batch_size != 0 else 0

    choice_cluster = torch.zeros([N]).cuda()
    for _ in range(max_iter):
        for bn in range(batchs + last):
            if bn == batchs and last == 1:
                _end = -1
            else:
                _end = (bn + 1) * batch_size
            X_batch = X[bn * batch_size: _end]

            dis_batch = euclidean_metric_gpu(X_batch, init_centers)
            choice_cluster[bn * batch_size: _end] = torch.argmin(dis_batch, dim=1)

        init_centers_pre = init_centers.clone()
        for index in range(K):
            selected = torch.nonzero(choice_cluster == index).squeeze().cuda()
            selected = torch.index_select(X, 0, selected)
            init_centers[index] = selected.mean(dim=0)

        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((init_centers - init_centers_pre) ** 2, dim=1)
            ))
        if center_shift < tol:
            break

    k_mean = init_centers.detach().cpu().numpy()
    choice_cluster = choice_cluster.detach().cpu().numpy()
    return k_mean, choice_cluster

if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import time

    n = 500000
    n1 = 5000
    K = 5
    data, label = make_blobs(n_samples=n, n_features=3, centers=K)
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(1, 3, 1, projection='3d', facecolor='white')
    ax.scatter(data[:n1, 0], data[:n1, 1], data[:n1, 2], c=label[:n1])
    ax.set_title("Data")

    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=K, max_iter=1000, tol=1e-40)
    st = time.time()
    model.fit_transform(data, label)
    et = time.time()
    print(f"Sklearn KMeans fitting time: {(et-st):.3f}ms")
    sk_pred_label = model.predict(data)
    ax = fig.add_subplot(1, 3, 2, projection='3d', facecolor='white')
    ax.scatter(data[:n1, 0], data[:n1, 1], data[:n1, 2], c=sk_pred_label[:n1])
    ax.set_title(f"S-KM:{(et-st):.1f}ms")

    X = torch.from_numpy(data.astype(np.float32)).cuda()
    st = time.time()
    mean, pre_label = kmeans_fun_gpu(X, K=K, max_iter=1000, batch_size=8096, tol=1e-40)
    et = time.time()
    print(f"KMeans-Batch-pytorch fitting time: {(et-st):.3f}ms")
    ax = fig.add_subplot(1, 3, 3, projection='3d', facecolor='white')
    ax.scatter(data[:n1, 0], data[:n1, 1], data[:n1, 2], c=pre_label[:n1])
    ax.set_title(f"KM-B:{(et-st):.1f}ms")

    plt.show()