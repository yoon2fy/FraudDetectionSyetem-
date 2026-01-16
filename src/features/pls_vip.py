from sklearn.cross_decomposition import PLSRegression
import numpy as np

def calculate_vip(pls, X):
    t = pls.x_scores_
    w = pls.x_weights_
    q = pls.y_loadings_

    p, h = w.shape
    vip = np.zeros((p,))
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)

    for i in range(p):
        vip[i] = np.sqrt(p * np.sum((w[i, :]**2) * s.flatten()) / total_s)

    return vip
