import pandas as pd
import numpy as np
import math
from datetime import datetime
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

path = 'data/'
scoring = 'neg_mean_absolute_error'

def read_data():
    x_train = pd.read_csv(path + 'train.csv', index_col='Id')
    x_test = pd.read_csv(path + 'test.csv', index_col='Id')

    x_train.dropna(axis=0, subset=['SalePrice'], inplace=True)

    y_train = x_train.SalePrice
    x_train.drop(['SalePrice'], axis=1, inplace=True)
    return x_train, y_train, x_test


def read_preprocessed_data(postfix):
    x_train = pd.read_csv(path + f'xtrain_{postfix}.csv')
    y_train = pd.read_csv(path + f'ytrain_{postfix}.csv')
    x_test = pd.read_csv(path + f'xtest_{postfix}.csv')
    return x_train, y_train, x_test


def write_preprocessed_data(postfix, x_train, y_train, x_test):
    x_train.to_csv(path + f'xtrain_{postfix}.csv')
    y_train.to_csv(path + f'ytrain_{postfix}.csv')
    x_test.to_csv(path + f'xtest_{postfix}.csv')


def get_cv_score(x, y, model, **fit_params):
    return -cross_val_score(model, x, y, cv=5, fit_params=fit_params,
                            scoring=scoring).mean()


def grid_search(x, y, model, params):
    gs = GridSearchCV(model, param_grid=params, scoring=scoring)
    gs.fit(x, y)
    return gs.best_score_, gs.best_params_


def randomized_search(x, y, model, params, n_iter=50, fit_params={}):
    best = [[1e9, 0]]
    while True:
        print(datetime.now())
        random_search = RandomizedSearchCV(model, param_distributions=params,
            n_iter=n_iter, cv=5, verbose=0, scoring=scoring)
        random_search.fit(x, y, **fit_params)
        best_score = -random_search.best_score_
        best_params = random_search.best_params_
        if best[-1][0] > best_score:
            best.append([best_score, best_params])
            best = sorted(best, key=lambda x: x[0])[:10]
        print('local best', best_score, best_params)
        print('global best', *best[0])
