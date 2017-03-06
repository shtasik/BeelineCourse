import numpy
from sklearn import svm

import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
import random

data = pd.read_csv('data.csv')

X = data.drop(['id', 'diagnosis'], axis=1).values
y = data['diagnosis'].map({'B': 1, 'M': 0}).values
y_categorical = np_utils.to_categorical(y, 2)

scores_f = []
scores_recall = []
scores_presision = []

kfold = StratifiedKFold(n_splits=10, shuffle=True)

for train, test in kfold.split(X, y):
    scaler = MinMaxScaler()
    t = np.size(train)
    scaler.fit(np.log(X[train] + 1))

#1
    bagging_train = []
    for i in range (0, t):
        bagging_train.append(random.choice(train))
    X_train = pd.DataFrame(scaler.transform(np.log(X[bagging_train] + 1))).values
    X_test = pd.DataFrame(scaler.transform(np.log(X[test] + 1))).values

    y_pred = numpy.zeros(len(X_test))

    params_xgb = {'n_estimators': range(30, 76, 5), 'learning_rate': [0.5, 0.4, 0.3],
                  'max_depth': range(2, 4, 1), 'reg_lambda': [1, 1.1, 1.2, 1.3], 'reg_alpha': [0.05, 0.01, 0.1],
                  'gamma': [0, 2]}
    grid_search_xgb = GridSearchCV(XGBClassifier(), params_xgb)

    grid_search_xgb.fit(X_train, y[bagging_train])
    y_pred_xgb = grid_search_xgb.best_estimator_.predict(X_test)
    y_pred = map(sum, zip(y_pred_xgb, y_pred))
    print ("\nXGB:")
    print grid_search_xgb.best_params_
    print f1_score(y[test], y_pred_xgb)
    print recall_score(y[test], y_pred_xgb)
    print precision_score(y[test], y_pred_xgb)

#2
    bagging_train = []
    for i in range (0, t):
        bagging_train.append(random.choice(train))
    X_train = pd.DataFrame(scaler.transform(np.log(X[bagging_train] + 1))).values

    params_ada = {'n_estimators': range(10, 56, 5), 'learning_rate': [0.01, 0.01, 0.5, 0.9, 1, 1.1, 1,45, 1.5, 1.55, 1.6, 1.7, 2]}
    grid_search_ada = GridSearchCV(AdaBoostClassifier(), params_ada)
    grid_search_ada.fit(X_train, y[bagging_train])
    y_pred_ada = grid_search_ada.best_estimator_.predict(X_test)
    y_pred = map(sum, zip(y_pred_ada, y_pred))
    print ("\nAdaBoost:")
    print grid_search_ada.best_params_
    print f1_score(y[test], y_pred_ada)
    print recall_score(y[test], y_pred_ada)
    print precision_score(y[test], y_pred_ada)

#3
    bagging_train = []
    for i in range (0, t):
        bagging_train.append(random.choice(train))
    X_train = pd.DataFrame(scaler.transform(np.log(X[bagging_train] + 1))).values

    params_random = {'n_estimators': range(15, 56, 5), 'max_depth' : range(5, 41, 5)}
    grid_search_random = GridSearchCV(RandomForestClassifier(), params_random)
    grid_search_random.fit(X_train, y[bagging_train])
    y_pred_rf = grid_search_random.best_estimator_.predict(X_test)
    y_pred = map(sum, zip(y_pred_rf, y_pred))
    print ("\nRF:")
    print grid_search_random.best_params_
    print f1_score(y[test], y_pred_rf)
    print recall_score(y[test], y_pred_rf)
    print precision_score(y[test], y_pred_rf)

#4
    bagging_train = []
    for i in range (0, t):
        bagging_train.append(random.choice(train))
    X_train = pd.DataFrame(scaler.transform(np.log(X[bagging_train] + 1))).values

    params_svm = {'kernel' :['poly', 'rbf', 'linear'], 'gamma': [0.01, 0.05, 0.5, 0.6, 0.7, 0.8, 1.45, 1.5, 1.55, 1.6,
                                                                 1.65, 1.7, 1.75, 1.8, 1.9],
                  'C': [0.01, 0.03, 0.05, 0.1, 0.4, 0.5, 0.6, 1, 1.4, 1.5, 1.6, 2, 2.5]}

    grid_search_svm = GridSearchCV(svm.SVC(), params_svm)
    grid_search_svm.fit(X_train, y[bagging_train])
    y_pred_svc = grid_search_svm.best_estimator_.predict(X_test)
    y_pred = map(sum, zip(y_pred_svc, y_pred))
    print ("\nSVM:")
    print grid_search_svm.best_params_
    print f1_score(y[test], y_pred_svc)
    print recall_score(y[test], y_pred_svc)
    print precision_score(y[test], y_pred_svc)

#5
    bagging_train = []
    for i in range (0, t):
        bagging_train.append(random.choice(train))
    X_train = pd.DataFrame(scaler.transform(np.log(X[bagging_train] + 1))).values

    model_seq = Sequential()
    model_seq.add(Dense(output_dim=600, input_shape=(X_train.shape[1], ), activation='relu', init='lecun_uniform'))
    model_seq.add(Dropout(0.6))
    model_seq.add(Dense(output_dim=250, activation='relu', init='lecun_uniform'))
    model_seq.add(Dropout(0.4))
    model_seq.add(Dense(output_dim=2, activation='softmax', init='lecun_uniform'))
    model_seq.compile(loss='binary_crossentropy', optimizer='adam')
    model_seq.fit(X_train, y_categorical[bagging_train], nb_epoch=11, verbose=2)
    y_pred_seq = model_seq.predict_classes(X_test)
    y_pred = map(sum, zip(y_pred_seq, y_pred))
    print ("\nNeural:")
    print f1_score(y[test], y_pred_seq)
    print recall_score(y[test], y_pred_seq)
    print precision_score(y[test], y_pred_seq)

    y_pred = map(lambda x: 1 if ((x / 5.) > 0.5) else 0, y_pred)

    print ("\nBagging:")
    scores_f.append(f1_score(y[test], y_pred))
    print f1_score(y[test], y_pred)
    scores_recall.append(recall_score(y[test], y_pred))
    print recall_score(y[test], y_pred)
    scores_presision.append(precision_score(y[test], y_pred))
    print precision_score(y[test], y_pred)

print('\nf1:', np.array(scores_f).mean())
print('recall:', np.array(scores_recall).mean())
print('precision:', np.array(scores_presision).mean())