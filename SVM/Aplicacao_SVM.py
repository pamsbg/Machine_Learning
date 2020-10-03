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
from numba import jit, cuda
# to measure exec time
from timeit import default_timer as timer
import time
import inspect, pylab
# from sklearn.externals import joblib

# def SalvarModelo(my_model):
#     joblib.dump(my_model, "my_model.pkl") 
#     # and later... 
#     my_model_loaded = joblib.load("my_model.pkl")



def preparacaodados(lags):
    dateparse=lambda dates: pd.datetime.strptime(dates, '%Y-%m')
    dados = pd.read_csv('Matriz_vazao_regress.csv', sep=';', parse_dates=['Month'], index_col='Month', date_parser =dateparse)
    
    dados_totais = dados.iloc[:, 13:14].values
    
    tempo = np.array(dados.index[312:].tolist(),dtype='datetime64[D]')
    tempo=tempo.reshape(-1,1)
    X_tempo ,y_tempo = prepare_data(tempo,lags)
    scaler = preprocessing.StandardScaler()
    dados_escalados = scaler.fit_transform(dados_totais)
    treino = dados.iloc[:312, 13:14].values
    treino_escalados = scaler.fit_transform(treino)
    teste = dados.iloc[312:432, 13:14].values
    teste_escalados = scaler.fit_transform(teste)
    validacao = dados.iloc[346:432, 13:14]

    
    X_total,y_total = prepare_data(dados_escalados,lags)
    X_train,y_train = prepare_data(treino_escalados,lags)
    X_test,y_test = prepare_data(teste_escalados,lags)
    y_true = y_test
    
    plt.plot(y_test, label='Dados Originais de Vazão | y ou t+1', color ='blue')    
    plt.plot(X_test, label='Dados Passados | X ou t', color='orange')
    plt.legend(loc='upper left')
    plt.title('Dados passados em um período')
    plt.show()
    return X_train, y_train,X_test,y_test, X_total,y_total, scaler, X_tempo, y_tempo


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
    
def RodarSVM(X_train, y_train, X_test, y_test, scaler, lags, X_tempo, y_tempo):
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
    
    regressor_linear = SVR(kernel='rbf', C=1.5, epsilon=0.5,gamma=1.0,verbose=1)
    # regressor_linear = SVR(kernel='rbf', verbose=1)
    print("Rodando Modelo")
    regressor_linear.fit(X_train, y_train)
    print("Criando Previsões")
    y_predict=regressor_linear.predict(X_test)
    y_test = scaler.inverse_transform(y_test)
    y_predict = scaler.inverse_transform(y_predict)
    X_test = scaler.inverse_transform(X_test)
    
    
    
    print("Calculando Pearson")
    pearson = pearsonr(y_test, y_predict)
    print("pearson:" + str(pearson))
    r2 = r2_score(y_test, y_predict)
    print("r2:" + str(r2))
    mse= mean_squared_error(y_test,y_predict)
    print("mse:" + str(mse))
    mae= mean_absolute_error(y_test,y_predict)
    print("mae:" + str(mae))
    
    # alteração de variável para criação de gráficos, adicionando data
    y_predict=y_predict.reshape(-1,1)
    df_y_predict=pd.DataFrame(y_predict)
    
    y_tempo = y_tempo.reshape(-1,1)
    df_y_tempo = pd.DataFrame(y_tempo, dtype='datetime64[D]')
    
    y_predict=df_y_predict.set_index(df_y_tempo.iloc[:,0])
    
    X_test=X_test.reshape(-1,1)
    df_x_test=pd.DataFrame(X_test)
    
    X_tempo = X_tempo.reshape(-1,1)
    df_x_tempo = pd.DataFrame(X_tempo, dtype='datetime64[D]')
    
    X_test=df_x_test.set_index(df_x_tempo.iloc[:,0])
 
    
    print("Gerando Gráficos")
    # plt.scatter()
    # plt.scatter(X_test[0],y_test[0])
    
    plt.plot(X_test, label='Observado Lag ' + str(lags), color='orange')
    plt.plot(y_predict,label='Previsto', color='black')
    plt.legend(loc='best')
    plt.title('Resultados')
    plt.xlabel('Data', fontsize=14)
    plt.ylabel('Vazão', fontsize=14)    
    plt.savefig(('gráfico_svm_lag'+str(lags)+'.png'))
    plt.show()
    return r2, mse, mae

        
def CriarGraficos():
        # Quantidade de vendas para o Produto A
    valores_produto_A = [6,7,8,4,4]
    
    # Quantidade de vendas para o Produto B
    valores_produto_B = [3,12,3,4.1,6]
    
    # Cria eixo x para produto A e produto B com uma separação de 0.25 entre as barras
    x1 =  np.arange(len(valores_produto_A))
    x2 = [x + 0.25 for x in x1]
    
    # Plota as barras
    plt.bar(x1, valores_produto_A, width=0.25, label = 'Produto A', color = 'b')
    plt.bar(x2, valores_produto_B, width=0.25, label = 'Produto B', color = 'y')
    
    # coloca o nome dos meses como label do eixo x
    meses = ['Agosto','Setembro','Outubro','Novembro','Dezembro']
    plt.xticks([x + 0.25 for x in range(len(valores_produto_A))], meses)
    
    # inseri uma legenda no gráfico
    plt.legend()
    
    plt.title("Quantidade de Vendas")
    plt.show()
    
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
    return r2, mse,mae


def RodarModelos():
    
    t0 = time.time()
    # r2=[0.519414749386341,0.2938693108439203,0.43726183758534454,0.4743342229830483,0.33934437066068057,0.4184729169398904,0.3277643323055064,0.2985764009381916, 0.2753468193319707, 0.23508359585957805,0.2398079428261859,0.19006508466344663]
    maelist=[]
    r2list=[]
    mselist=[]
    maelistols=[]
    r2listols=[]
    mselistols=[]
    
    for lags in range(1, 13):
        print("Lag " + str(lags))
        X_train, y_train, X_test, y_test, X_total,y_total, scaler, X_tempo, y_tempo = preparacaodados(lags)
        # RodarSVMGridsearch(X_train, y_train, X_test, y_test,X_total,y_total, scaler)
        # print("SVM")
        r2, mse, mae=RodarSVM(X_train, y_train, X_test, y_test, scaler, lags, X_tempo, y_tempo)
        mselist.append(mse)
        r2list.append(r2)
        maelist.append(mae)
        # RodarLibSVM(X_train, y_train, X_test, y_test)
        # print("OLS")
        r2ols, mseols, maeols= OLS(X_train, y_train, X_test, y_test, scaler)
        mselistols.append(mseols)
        r2listols.append(r2ols)
        maelistols.append(maeols)
        # t1 = time.time() -t0   
        # print("tempo decorrido:" + str(t1))
    # r2concat = np.stack((r2list,r2listols), axis=1)    
    x1 = np.arange(1,13)
    x2 = [x + 0.25 for x in x1]
    plt.title("R-Squared")
    plt.bar(x1, r2list, width=0.25, label = 'R2 SVM', color = 'orange')
    plt.bar(x2, r2listols, width=0.25, label = 'R2 OLS', color = 'firebrick')
    plt.legend(loc='best')    
    plt.xlabel('Lag', fontsize=14)    
    plt.savefig(('gráfico_svm_resumor2_lag.png'))
    plt.show()    
    
    plt.title("Mean Squared Error ")
    plt.bar(x1, mselist, width=0.25, label = 'MSE SVM', color = 'lawngreen')
    plt.bar(x2, mselistols, width=0.25, label = 'MSE OLS', color = 'darkolivegreen')
    plt.legend(loc='best')    
    plt.xlabel('Lag', fontsize=14)    
    
    plt.savefig(('gráfico_svm_resumomse_lag.png'))
    plt.show()    
    
    plt.title("Mean Absolut Error")
    plt.bar(x1, maelist, width=0.25, label = 'MAE SVM', color = 'deepskyblue')
    plt.bar(x2, maelistols, width=0.25, label = 'MAE OLS', color = 'mediumblue')
    plt.legend(loc='best')    
    plt.xlabel('Lag', fontsize=14)    
    plt.savefig(('gráfico_svm_resumomae_lag.png'))
    plt.show()    
    # t1 = time.time() - t0   
    # print("tempo decorrido total:" + str(t1))