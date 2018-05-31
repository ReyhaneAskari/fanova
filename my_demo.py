import sys
sys.path.append('/u/askarihr/repos/fanova/')

import numpy as np
import fanova
import fanova.visualizer as viz
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

X_full = np.loadtxt('/u/askarihr/repos/fanova/examples/example_data/online_lda/online_lda_features.csv', delimiter=',')
y_full = np.loadtxt('/u/askarihr/repos/fanova/examples/example_data/online_lda/online_lda_responses.csv', delimiter=',')

n_samples = 128

indices = np.random.choice(X_full.shape[0], n_samples)

if n_samples < X_full.shape[0]:
    X = X_full[indices]
    y = y_full[indices]
else:
    X = X_full
    y = y_full
f = fanova.fANOVA(X, y, n_trees=32, bootstrapping=True)

for i in range(3):

    gt = []

    unique_values = list(set(X_full[:, i]))
    unique_values.sort()

    for v in unique_values:
        indices = np.where(X_full[:, i] == v)
        gt.append((v, np.mean(y_full[indices]), np.var(y_full[indices])))

    gt = np.array(gt)

    plt.figure()
    mew = np.linspace(np.min(X[:, i]), np.max(X[:, i]), 100)
    mew2 = np.array([f.marginal_mean_variance_for_values([i], [v]) for v in mew])

    m = mew2[:, 0]
    s = np.sqrt(mew2[:, 1])

    plt.plot(mew, m)
    plt.fill_between(mew, m - s, m + s, alpha=.3)

    plt.scatter(gt[:, 0], gt[:, 1])
    plt.ylabel('Perplexity')
    plt.xlabel('parameter {}'.format(i))

    plt.savefig("fANOVA__" + str(i) + ".png")
