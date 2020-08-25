# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from sklearn.svm import SVR
import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import pearsonr
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from numba import jit, cuda
# to measure exec time
from timeit import default_timer as timer
import time

def preparacaodados(lags):
    dados = pd.read_csv("Matriz_vazao_regress.csv", sep=';')    
    dados_totais = dados.iloc[:, 14:15].values
    dados_escalados = preprocessing.scale(dados_totais)
    treino = dados.iloc[:260, 14:15].values
    treino_escalados = preprocessing.scale(treino)
    teste = dados.iloc[261:346, 14:15].values
    teste_escalados = preprocessing.scale(teste)
    validacao = dados.iloc[346:432, 14:15]

    y_predict=[]
    X_total,y_total = prepare_data(dados_escalados,lags)
    X_train,y_train = prepare_data(treino_escalados,lags)
    X_test,y_test = prepare_data(teste_escalados,lags)
    y_true = y_test
    
    plt.plot(y_test, label='Dados Originais de Vazão | y ou t+1', color ='blue')    
    plt.plot(X_test, label='Dados Passados | X ou t', color='orange')
    plt.legend(loc='upper left')
    plt.title('Dados passados em um período')
    plt.show()
    return X_train, y_train,X_test,y_test, X_total,y_total


def prepare_data(data, lags):
    X,y = [],[]
    for row in range(len(data)-lags-1):
        a = data[row:(row+lags),0]            
        X.append(a)
        y.append(data[row-lags,0])
    return np.array(X),np.array(y)

def mycustomscorer(y_test, prediction):
    mycustomscorer, _ = pearsonr(y_test, prediction)
    return mycustomscorer

def RodarSVMGridsearch(X_train, y_train, X_test, y_test,X_total,y_total):
        # usando SVR simples
    # regressor_linear = SVR(kernel='linear')
    # usando gridsearch
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    Cs=np.arange(1,10,0.1).tolist()
    gammas = np.arange(0.001,3,0.01).tolist()
    degrees = np.arange(0,3,1).tolist()
    regressor_linear = svr = GridSearchCV(SVR(),
                                param_grid={
                                    "kernel": ['rbf'],
                                    "C":Cs,
                                    "gamma":[0.0001,0.001,0.01,0.1,1,1.5],
                                    # 'degree':degrees,
                                     "epsilon":[0.1,0.5,1],
                                   
                                    }, verbose=1, n_jobs=(6), cv=25, scoring=scorer)
    print("Rodando Modelo")
    regressor_linear.fit(X_total, y_total)
    print("Criando Previsões")
    y_predict=regressor_linear.predict(X_test)
    
    print("Calculando Pearson")
    pearson = pearsonr(y_test, y_predict)
    print("pearson:" + str(pearson))
    r2 = r2_score(y_test, y_predict)
    print("r2:" + str(r2))
    # print("Gerando Gráficos")
    
    # plt.scatter(X_test,y_test)
    # plt.plot(X_test, regressor_linear.predict(X_test), color='red')
    
    sv_ratio = regressor_linear.best_estimator_.support_.shape[0] / X_train.size
    print("Support vector ratio: %.3f" % sv_ratio)
    print("Best parameters set found on development set:")
    print()
    print(regressor_linear.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = regressor_linear.cv_results_['mean_test_score']
    stds = regressor_linear.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, regressor_linear.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    
def RodarSVM(X_train, y_train, X_test, y_test):
    # {'C': 1.5, 'epsilon': 0.1, 'gamma': 0.1, 'kernel': 'rbf'} r2 0.55
    
    # {'C': 1.5, 'epsilon': 0.5, 'gamma': 1, 'kernel': 'rbf'}0,60
    # {'C': 1.5, 'epsilon': 0.1, 'gamma': 1.5, 'kernel': 'rbf'}0,62
    # {'C': 1.5, 'epsilon': 0.5, 'gamma': 1.5, 'kernel': 'rbf'}0,60
    # {'C': 1, 'epsilon': 0.5, 'gamma': 1.5, 'kernel': 'rbf'}0,60
    # {'C': 1.5, 'epsilon': 0.1, 'gamma': 0.1, 'kernel': 'rbf'}0,55
    # {'C': 1, 'epsilon': 0.1, 'gamma': 0.0001, 'kernel': 'linear'}0,49
    # {'C': 1.5, 'epsilon': 0.5, 'gamma': 0.0001, 'kernel': 'linear'}0,47
    # {'C': 0.01, 'epsilon': 0.5, 'gamma': 0.0001, 'kernel': 'linear'}0,50 lag 7, 0,39 lag 1
    # {'C': 1, 'epsilon': 0.5, 'gamma': 0.0001, 'kernel': 'linear'}0,47
    # {'C': 0.1, 'epsilon': 0.5, 'gamma': 0.0001, 'kernel': 'linear'}0,47
    
    regressor_linear = SVR(kernel='rbf', C=30.5, epsilon=0.001,gamma=1.5, verbose=1)
    # regressor_linear = SVR(kernel='rbf', verbose=1)
    print("Rodando Modelo")
    regressor_linear.fit(X_train, y_train)
    print("Criando Previsões")
    y_predict=regressor_linear.predict(X_test)
    
    print("Calculando Pearson")
    pearson = pearsonr(y_test, y_predict)
    print("pearson:" + str(pearson))
    r2 = r2_score(y_test, y_predict)
    print("r2:" + str(r2))
    print("Gerando Gráficos")
    # plt.scatter()
    # plt.scatter(X_test[0],y_test[0])
    plt.plot(X_test, regressor_linear.predict(X_test), color='red')
    
def RodarModelos():
    
    t0 = time.time()
    for lags in range(1, 13):
        print("Lag " + str(lags))
        X_train, y_train, X_test, y_test, X_total,y_total = preparacaodados(lags)
        RodarSVMGridsearch(X_train, y_train, X_test, y_test,X_total,y_total)
        # RodarSVM(X_train, y_train, X_test, y_test)
        t1 = time.time() -t0   
        print("tempo decorrido:" + str(t1))
    t1 = time.time() -t0   
    print("tempo decorrido total:" + str(t1))

        


