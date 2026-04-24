# the basics
import os
import pandas as pd
import numpy as np

# project imports
from molecules_and_features import make_dataset

# basic sklearn stuff
from sklearn import pipeline
from sklearn import model_selection
from xgboost import XGBRegressor
from sklearn import cross_decomposition

# preprocessing/data selection
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

#supervised ml
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression


DATA_DIR = os.getenv('NICEATM_ACUTE_ORAL_DATA')


def get_regress_stats(model, X, y):
    """

    :param model: If None, assume X == y_true and y == y_pred, else should be a trained model
    :param X: Data to predict
    :param y: correct classes
    :return:
    """
    if not model:
        predictions = y
        y = X
    else:
        if 'predict_classes' in dir(model):
            predictions = model.predict(X, verbose=0)[:, 0]
        else:
            predictions = model.predict(X)

    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)

    return {'MAE': mae, 'r2': r2}


def split_train_test(X, y, n_split, test_set_size, seed, major_subsample=None):
    """ Splits data into training and test sets

    :param n_split:
    :param test_set_size:
    :param seed:
    :return:
    """
    assert X.shape[0] == y.shape[0], 'The lengths of X and y do not match X == {}, y == {}.'.format(X.shape[0],
                                                                                                    y.shape[0])

    if major_subsample != None:
        if sum(y) > y.shape[0] / 2:
            major_class = 1
            minor_class = 0
        else:
            major_class = 0
            minor_class = 1

        major_class_index = y[y == major_class].index
        major_class_index_remove = np.random.choice(np.array(major_class_index),
                                                    int(major_class_index.shape[0] * (1. - major_subsample)),
                                                    replace=False)
        X.drop(X.index[major_class_index_remove], inplace=True)
        y.drop(y.index[major_class_index_remove], inplace=True)


    # split X into training sets and test set the size of test_set_
    if test_set_size != 0:
        batch_size = int(y.shape[0] * (1 - test_set_size) // n_split)  # calculating batch size
        train_size = int(batch_size * n_split)

        X_train_tmp, X_test, y_train_class_tmp, y_test_class = model_selection.train_test_split(X,
                                                                                                y,
                                                                                                train_size=train_size,
                                                                                                random_state=seed)
    else:
        X_train_tmp = X
        y_train_class_tmp = y
        X_test = None
        y_test_class = None

    cv = model_selection.KFold(shuffle=True, n_splits=n_split, random_state=seed)
    valid_idx = []  # indexes for new train dataset
    for (_, valid) in cv.split(X_train_tmp, y_train_class_tmp):
        valid_idx += valid.tolist()

    X_train = X_train_tmp.iloc[valid_idx]
    y_train_class = y_train_class_tmp.iloc[valid_idx]

    return X_train, y_train_class, X_test, y_test_class

# ALGS is a list of tuples
# where item 1 is the name
# item 2 is a scikit-learn machine learning regressor
# and item 3 is the paramaters to grid search through

seed = 0


REGRESSOR_ALGS = [
    ('rfr', RandomForestRegressor(max_depth=10, # max depth 10 to prevent overfitting
                                  random_state=seed), {'rfr__n_estimators':[5, 10, 25,50,100]}),
    ('knnr', KNeighborsRegressor(metric='euclidean'), {'knnr__n_neighbors':[1, 3, 5,7,10],
                                                        'knnr__weights': ['uniform', 'distance']}),
    ('svr', SVR(), {'svr__kernel': ['rbf'],
                                     'svr__gamma': [1e-2, 1e-3],
                                     'svr__C': [1, 10]}),
    ('xgb', XGBRegressor(random_state=seed), {}),

]