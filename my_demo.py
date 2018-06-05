import sys
sys.path.append('/u/askarihr/repos/fanova/')

import numpy as np
from numpy import array
import pickle
import fanova
import fanova.visualizer as viz
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

dataset_name = 'B.1'


def plot_bars(importance_list):
    objects = []
    performance = []
    for item in importance_list:
        print(str(item))
        objects.append(item[0])
        performance.append(item[1])
    y_pos = np.arange(len(objects))
    plt.figure()
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.xticks(rotation=-45, ha='left')
    plt.ylabel('importance')
    plt.savefig('outs/' + dataset_name + '/bar_plot.png', bbox_inches="tight")

with open('data/' + dataset_name + '.pkl', 'rb') as f:
    if dataset_name in ['A.1', 'B.1']:
        data = pickle.load(f, encoding='bytes')
    else:
        data = pickle.load(f)
y_results = []
x_params = []
x_param_names = []
num_features = len(data[0]['params'])

encoding = defaultdict(None)
encoding.default_factory = encoding.__len__

for i in range(len(data)):
    param = data[i]['params']
    x_param_names.append(
        [param[j]['name'] for j in range(num_features)])
    values = []
    for j in range(num_features):
        if param[j]['type'] == 'categorical':
            values.append(encoding[param[j]['value']])
        else:
            values.append(param[j]['value'])
    x_params.append(values)
    for result in data[i]['results']:
        if result['type'] == 'objective':
            y_results.append(result['value'])
            # there is only one result with objective as type
            break
X_full = array(x_params)
y_full = array(y_results)

n_samples = len(data)

indices = np.random.choice(X_full.shape[0], n_samples)

if n_samples < X_full.shape[0]:
    X = X_full[indices]
    y = y_full[indices]
else:
    X = X_full
    y = y_full

f = fanova.fANOVA(X, y, n_trees=32, bootstrapping=True)

importance_list = []
for i in range(num_features):

    gt = []
    label = x_param_names[0][i]
    importance_list.append(
        [label, f.quantify_importance((i, ))[(i,)]['individual importance']])
    unique_values = list(set(X_full[:, i]))
    unique_values.sort()

    for v in unique_values:
        indices = np.where(X_full[:, i] == v)
        gt.append((v, np.mean(y_full[indices]), np.var(y_full[indices])))

    gt = np.array(gt)

    plt.figure()
    mew = np.linspace(np.min(X[:, i]), np.max(X[:, i]), 100)
    mew2 = np.array(
        [f.marginal_mean_variance_for_values([i], [v]) for v in mew])

    m = mew2[:, 0]
    s = np.sqrt(mew2[:, 1])
    plt.plot(mew, m)
    plt.fill_between(mew, m - s, m + s, alpha=.3)

    # plt.scatter(gt[:, 0], gt[:, 1])
    plt.ylabel('Perplexity')
    plt.xlabel('{}'.format(label))

    plt.savefig('outs/' + dataset_name + '/' + str(i) + ".png")
plot_bars(importance_list)
