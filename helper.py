from time import time
import json
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_y(x):
    matrix = []
    for i in x:
        matrix.append(int(i))
    return np.array(matrix).reshape((81, 1)) - 1


def preprocess_x(x):
    matrix = []
    for i in x:
        matrix.append(int(i))
    return np.array(matrix).reshape((9, 9, 1)) / 9

def load_and_process(path, seed=5, test_portion=0.15):
    df = pd.read_csv(path)
    df['quizzes'] = df['quizzes'].apply(preprocess_x)
    df['solutions'] = df['solutions'].apply(preprocess_y)
    train_df, test_df = train_test_split(df, test_size=test_portion, random_state=seed)
    train_x, train_y, test_x, test_y = np.stack(train_df['quizzes'].values), np.stack(train_df['solutions'].values), \
                                       np.stack(test_df['quizzes'].values), np.stack(test_df['solutions'].values)
    return train_x, train_y, test_x, test_y


def plot(hist, by='loss', figsize=(15, 12),path='loss.png'):
    assert by in ['loss', 'acc']
    sns.set()
    hist = dict((k, v) for k, v in hist.items() if by in k)
    plt.figure(figsize=figsize)
    epochs = 0
    for k, v in hist.items():
        epochs = len(v)
        try:
            plt.plot(range(len(v)), v)
        except:
            v = [i.numpy() for i in v]
            plt.plot(range(len(v)), v)
    plt.ylabel(by)
    plt.xlabel('Epochs')
    plt.legend(list(hist.keys()))
    plt.xticks(range(0, epochs + 1, 1))
    plt.savefig()

def timeit(method):
    def timed(*args, **kw):
        ts = time()
        result = method(*args, **kw)
        te = time()
        print('{}_time'.format(method.__name__), (te - ts) / 60)
        return result

    return timed


def load_params(path_params):
    with open(path_params, 'r') as f:
        params = json.load(f)
    return params


def save_params(file, path_params):
    with open(path_params, 'w') as f:
        json.dump(file, f)
