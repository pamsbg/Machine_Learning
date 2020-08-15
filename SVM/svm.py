# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pdb as pdb
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV



dados = pd.read_csv("Matriz_vazao_regress.csv", sep=';')
treino = dados.iloc[:260, 14:15].values
teste = dados.iloc[261:346,14:15].values
validacao = dados.iloc[346:432,14:15]
# usando SVR simples
# regressor_linear = SVR(kernel='rbf')

# usando gridsearch
regressor_linear = svr = GridSearchCV(SVR(gamma=0.1),
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                               "kernel":['rbf', 'linear'],
                               "gamma": np.logspace(-2, 2, 5)})

y_predict=[]


for lags in range(1,13):

    print("Iniciando Loop, Lag:" + str(lags))
    def prepare_data(data, lags):
        X,y = [],[]
        for row in range(len(data)-lags-1):
            a = data[row:(row+lags),0]            
            X.append(a)
            y.append(data[row-lags,0])
        return np.array(X),np.array(y)
                      
    
        
    
    X_train,y_train = prepare_data(treino,lags)
    X_test,y_test = prepare_data(teste,lags)
    y_true = y_test
    
    plt.plot(y_test, label='Dados Originais de Vazão | y ou t+1', color ='blue')    
    plt.plot(X_test, label='Dados Passados | X ou t', color='orange')
    plt.legend(loc='upper left')
    plt.title('Dados passados em um período')
    plt.show()
    
    # X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1]))
    #X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1]))
    
    def mycustomscorer(y_test, prediction):
        mycustomscorer, _ = pearsonr(y_test, prediction)
        return mycustomscorer
    
    print("Criando função especial para cálculo de pearson")
    my_scorer = make_scorer(mycustomscorer, greater_is_better=True)
    
 
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
    
    sv_ratio = regressor_linear.best_estimator_.support_.shape[0] / treino.size
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
       
  