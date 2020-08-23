# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from sklearn.metrics import make_scorer
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
    
    treino = dados.iloc[:260, 14:15].values
    treino_escalados = preprocessing.scale(treino)
    teste = dados.iloc[261:346, 14:15].values
    teste_escalados = preprocessing.scale(teste)
    validacao = dados.iloc[346:432, 14:15]

    y_predict=[]
    X_train,y_train = prepare_data(treino,lags)
    X_test,y_test = prepare_data(teste,lags)
    y_true = y_test
    
    plt.plot(y_test, label='Dados Originais de Vazão | y ou t+1', color ='blue')    
    plt.plot(X_test, label='Dados Passados | X ou t', color='orange')
    plt.legend(loc='upper left')
    plt.title('Dados passados em um período')
    plt.show()
    return X_train, y_train,X_test,y_test


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

def RodarSVM(X_train, y_train, X_test, y_test):
        # usando SVR simples
    # regressor_linear = SVR(kernel='linear')
    # usando gridsearch
    Cs=np.arange(0.001,3,0.5).tolist()
    gammas = np.arange(0.001,1,0.1).tolist()
    regressor_linear = svr = GridSearchCV(SVR(),
                                param_grid={
                                    "kernel": ['linear'],
                                    "C":Cs,
                                    "gamma":gammas,
                                    # "epsilon":[0.001,0.01,0.1,0.5,1],
                                   
                                    }, verbose=1, n_jobs=(6), cv=10)
    print("Rodando Modelo")
    regressor_linear.fit(X_train, y_train)
    print("Criando Previsões")
    y_predict=regressor_linear.predict(X_test)
    
    print("Calculando Pearson")
    r2 = pearsonr(y_test, y_predict)
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
    
def RodarModelos():
    
    t0 = time.time()
    for lags in range(1, 13):
        print("Lag " + str(lags))
        X_train, y_train, X_test, y_test = preparacaodados(lags)
        RodarSVM(X_train, y_train, X_test, y_test)
        t1 = time.time() -t0   
        print("tempo decorrido:" + str(t1))
    t1 = time.time() -t0   
    print("tempo decorrido total:" + str(t1))

        


