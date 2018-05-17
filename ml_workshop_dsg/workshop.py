import math
import sklearn as sk
import pandas as pd
import numpy as np
import time
import random
import itertools
import math
from sklearn import preprocessing
from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.preprocessing import scale, normalize, StandardScaler
from scipy.spatial.distance import pdist


from sklearn.svm import SVC, LinearSVC, SVR, LinearSVR, OneClassSVM
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsClassifier, RadiusNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from matplotlib import pyplot as plt


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


def data_preprocessing(data):
    #Function prepare data to model

    # 1. Fill NA in TotalCharges column
    data['TotalCharges'] = data['TotalCharges'].fillna(0)
    print('Fill NA was finished!')

    # 2. Categorical predictors preprocessing
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    target = data.Churn.apply(lambda x: 1 if x == 'Yes' else 0).values
    df_categoric = data.select_dtypes(exclude=numerics).drop('Churn', axis=1)
    df_numeric = data.select_dtypes(include=numerics)

    df_categoric_bin = pd.get_dummies(df_categoric)
    print('Categorical predictors preprocessing was finished!')

    # 3. Scale numerical features
    scale = MinMaxScaler()
    df_numeric_scale = pd.DataFrame(scale.fit_transform(df_numeric),
                                    columns=df_numeric.columns,
                                    index=df_numeric.index)
    print('Scale numerical features was finished!')

    # 4. Drop high correlated features
    result = pd.concat([df_categoric_bin, df_numeric_scale], axis=1)
    corr_matrix = result.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    result.drop(to_drop, axis=1, inplace=True)

    print('Drop high correlated features was finished, we drop:', '\n'.join(to_drop))

    return result, target


def fit_model(algs, data, target, seed):

    Train, Test = train_test_split(data, test_size=0.2, random_state=10)
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=10)

    model = []
    acc_train_initial, acc_test_initial, roc_train, \
    roc_test, f1_train, f1_test, presicion_train, presicion_test = [], [], [], [], [], [], [], []

    training_time, prediction_time = [], []

    ts_start = time.time()

    for alg in algs:
        model.append(alg.__name__)

        if 'random_state' in alg().get_params().keys():
            clf = alg(random_state=seed)  # ,C=0.19)
        else:
            clf = alg()

        ts = time.time()
        # sw=compute_sample_weight('balanced', y=y_train)
        # sw[sw==sw.max()]=1000
        clf.fit(X_train, y_train)
        training_time.append(round(time.time() - ts, 4))

        ts = time.time()
        preds_train = clf.predict(X_train)
        preds_test = clf.predict(X_test)
        prediction_time.append(round(time.time() - ts, 5))

        # sw1=compute_sample_weight('balanced', y=y_test)
        acc_train_initial.append(round(accuracy_score(y_train, preds_train), 4))
        acc_test_initial.append(round(accuracy_score(y_test, preds_test), 4))
        roc_train.append(round(roc_auc_score(y_train, preds_train) * 100, 4))
        roc_test.append(round(roc_auc_score(y_test, preds_test) * 100, 4))
        f1_train.append(f1_score(y_train, preds_train))
        f1_test.append(f1_score(y_test, preds_test))
        presicion_train.append(precision_score(y_train, preds_train))
        presicion_test.append(precision_score(y_test, preds_test))

        print('Tested {} in {:.2f} seconds'.format(alg.__name__, time.time() - ts_start))

    model = [x + '_default' for x in model]

    models = pd.DataFrame({
        'MODEL': model,
        'TRAIN_ACC': acc_train_initial,
        'TEST_ACC': acc_test_initial,
        'TRAIN_AUC': roc_train,
        'TEST_AUC': roc_test,
        'TRAIN_F1': f1_train,
        'TEST_F1': f1_test,
        'TRAIN_PRES': presicion_train,
        'TEST_PRES': presicion_test,
        'TRAINING_TIME': training_time,
        'PREDICTION_TIME': prediction_time})

    return Train, Test, models

def fit_model_tune(algs, data, target, search_space, models):

    Train, Test = train_test_split(data, test_size=0.2, random_state=10)
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=10)

    best_models = {}
    model = []
    clfs = []
    acc_train_tuned, acc_test_tuned, roc_train, \
    roc_test, f1_train, f1_test, presicion_train, presicion_test = [], [], [], [], [], [], [], []
    tuning_time, training_time, prediction_time = [], [], []

    X_all = data
    y_all = target
    cv_fold = np.hstack([np.zeros((y_all.shape[0] // 2)), np.ones((y_all.shape[0] // 2))])
    ps = PredefinedSplit(cv_fold)


    for alg in algs:
        ts_start = time.time()
        model.append(alg.__name__)
        ts = time.time()
        if alg != LogisticRegression:
            clf = GridSearchCV(alg(), search_space[alg.__name__], scoring='precision_weighted', cv=3,
                               return_train_score=True, verbose=2, n_jobs=-1)
        else:
            clf = GridSearchCV(alg(), search_space[alg.__name__], scoring='precision_weighted', cv=3,
                               return_train_score=True, verbose=2)
        # sw=compute_sample_weight('balanced', y=y_all)
        clf.fit(X_all, y_all)
        tuning_time.append(round(time.time() - ts, 4))
        clfs.append(clf)

        ts = time.time()
        clf = alg(**clf.best_params_)
        clf.fit(X_train, y_train)
        best_models[alg.__name__] = clf

        training_time.append(round(time.time() - ts, 4))

        ts = time.time()
        preds_train = clf.predict(X_train)
        preds_test = clf.predict(X_test)
        prediction_time.append(round(time.time() - ts, 5))

        acc_train_tuned.append(round(accuracy_score(y_train, preds_train), 4))
        acc_test_tuned.append(round(accuracy_score(y_test, preds_test), 4))
        roc_train.append(round(roc_auc_score(y_train, preds_train) * 100, 4))
        roc_test.append(round(roc_auc_score(y_test, preds_test) * 100, 4))
        f1_train.append(f1_score(y_train, preds_train))
        f1_test.append(f1_score(y_test, preds_test))
        presicion_train.append(precision_score(y_train, preds_train))
        presicion_test.append(precision_score(y_test, preds_test))

        print('\n\nTuned {} in {:.2f} seconds\n\n'.format(alg.__name__, time.time() - ts_start))

    model = [x + '_tuned' for x in model]

    models = models.append(
        pd.DataFrame({
            'MODEL': model,
            'TRAIN_ACC': acc_train_tuned,
            'TEST_ACC': acc_test_tuned,
            'TRAIN_AUC': roc_train,
            'TEST_AUC': roc_test,
            'TRAIN_F1': f1_train,
            'TEST_F1': f1_test,
            'TRAIN_PRES': presicion_train,
            'TEST_PRES': presicion_test,
            'TRAINING_TIME': training_time,
            'PREDICTION_TIME': prediction_time}))

    return Train, Test, models, best_models