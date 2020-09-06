# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 20:00:05 2020

@author: pamsb
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
from scipy.stats import pearsonr
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# to measure exec time
from timeit import default_timer as timer
import time

# from sklearn.externals import joblib

# def SalvarModelo(my_model):
#     joblib.dump(my_model, "my_model.pkl") 
#     # and later... 
#     my_model_loaded = joblib.load("my_model.pkl")

def ReamostrarDados(array):
        matriz = []
        
        
        dia = 1
        diainicio =0
        diasdecorridos=24
        # tamanhoarray=(array.size//24)
        # matriz = np.zeros(int(tamanhoarray)
        for d in range(0,1460):
            subset = array[diainicio:diasdecorridos]
            soma=0
            contador=0
            diainicio=diainicio+24
            diasdecorridos=diasdecorridos+24
            for valor in subset:
                contador = contador +1
                soma = soma + (valor)
                media = soma/contador           
            
            matriz.append(media)            
            

            
        return matriz

def preparacaodados(lags):
    treino = pd.read_csv("TrainData_Blind.txt")    
    treino = treino.to_numpy()
    matriz = ReamostrarDados(treino)
    teste = pd.read_csv("TestData_Blind.txt")    
    scaler = preprocessing.StandardScaler()
    
    
    treino_escalados = scaler.fit_transform(treino)
    
    teste_escalados = scaler.fit_transform(teste)
    

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
    return X_train, y_train,X_test,y_test, X_total,y_total, scaler


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

def RodarSVMGridsearch(X_train, y_train, X_test, y_test,X_total,y_total, scaler):
        # usando SVR simples
    # regressor_linear = SVR(kernel='linear')
    # usando gridsearch
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    Cs=np.arange(0.1,10,0.1).tolist()
    gammas = np.arange(0.001,3,0.001).tolist()
    degrees = np.arange(0,3,1).tolist()
    epsilons = np.arange(0.001,1,0.00001)
    regressor_linear = svr = GridSearchCV(SVR(),
                                param_grid={
                                    "kernel": ['rbf', 'sigmoid'],
                                      "C":Cs,
                                    "gamma":[0.1,0.001,0.0001,0.00001,1,1.5,0.5,2,2.5,3],
                                    # 'degree':degrees,
                                      "epsilon":[0.1, 0.0001,0.00001,0.1],
                                   
                                    }, verbose=1, n_jobs=(6), cv=25, scoring=scorer)
    
    
    # parameters = {
    #      'kernel': ('linear', 'rbf','sigmoid'), 
    #      'C': np.arange(1, 40,0.5).tolist(), 
    #      'gamma': [0.1,0.0001,0.00001,0.1,1.5,1.0],
    #      'epsilon':[0.1, 0.0001,0.00001,0.1]
    #      }
    # regressor_linear=RandomizedSearchCV(SVR(), parameters, random_state=4, n_iter=100000, n_jobs=(6), scoring=scorer)
    print("Rodando Modelo")
    regressor_linear.fit(X_total, y_total)
    print("Criando Previsões")
    y_predict=regressor_linear.predict(X_test)
    y_test = scaler.inverse_transform(y_test)
    y_predict = scaler.inverse_transform(y_predict)
    print("Calculando Pearson")
    pearson = pearsonr(y_test, y_predict)
    print("pearson:" + str(pearson))
    r2 = r2_score(y_test, y_predict)
    print("r2:" + str(r2))
    mse= mean_squared_error(y_test,y_predict)
    print("mse:" + str(mse))
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
    # for mean, std, params in zip(means, stds, regressor_linear.cv_results_['params']):
    #     print("%0.3f (+/-%0.03f) for %r"
    #           % (mean, std * 2, params))
    # print()
    
def RodarSVM(X_train, y_train, X_test, y_test, scaler):
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
    
    regressor_linear = SVR(kernel='rbf', C=9.1, epsilon=0.0001,gamma=0.1, verbose=1)
    # regressor_linear = SVR(kernel='rbf', verbose=1)
    print("Rodando Modelo")
    regressor_linear.fit(X_train, y_train)
    print("Criando Previsões")
    y_predict=regressor_linear.predict(X_test)
    y_test = scaler.inverse_transform(y_test)
    y_predict = scaler.inverse_transform(y_predict)
    print("Calculando Pearson")
    pearson = pearsonr(y_test, y_predict)
    print("pearson:" + str(pearson))
    r2 = r2_score(y_test, y_predict)
    print("r2:" + str(r2))
    mse= mean_squared_error(y_test,y_predict)
    print("mse:" + str(mse))
    mae= mean_absolute_error(y_test,y_predict)
    print("mae:" + str(mae))
    print("Gerando Gráficos")
    # plt.scatter()
    # plt.scatter(X_test[0],y_test[0])
    plt.plot(X_test, regressor_linear.predict(X_test), color='red')
    


        

def OLS(X_train, y_train, X_test, y_test, scaler):
    
    reg = linear_model.LinearRegression()
    reg.fit(X_train, y_train)
    reg_predict = reg.predict(X_test)
    
    y_test = scaler.inverse_transform(y_test)
    reg_predict = scaler.inverse_transform(reg_predict)
    print("Calculando Pearson")
    pearson = pearsonr(y_test, reg_predict)
    print("pearson:" + str(pearson))
    r2 = r2_score(y_test, reg_predict)
    print("r2:" + str(r2))
    
    mse= mean_squared_error(y_test,reg_predict)
    print("mse:" + str(mse))
    mae= mean_absolute_error(y_test,reg_predict)
    print("mae:" + str(mae))


def RodarModelos():
    
    t0 = time.time()
    for lags in range(1, 13):
        print("Lag " + str(lags))
        X_train, y_train, X_test, y_test, X_total,y_total, scaler = preparacaodados(lags)
        # RodarSVMGridsearch(X_train, y_train, X_test, y_test,X_total,y_total, scaler)
        # print("SVM")
        RodarSVM(X_train, y_train, X_test, y_test, scaler)
        # RodarLibSVM(X_train, y_train, X_test, y_test)
        # print("OLS")
        # OLS(X_train, y_train, X_test, y_test, scaler)
        # t1 = time.time() -t0   
        # print("tempo decorrido:" + str(t1))
    # t1 = time.time() - t0   
    # print("tempo decorrido total:" + str(t1))