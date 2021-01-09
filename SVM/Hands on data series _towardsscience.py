# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 19:07:57 2020

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
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=15,6
from sklearn import linear_model
import numpy as np
from scipy.stats import pearsonr
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from numba import jit, cuda
# to measure exec time
from timeit import default_timer as timer
import time
import inspect, pylab
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
plt.style.use('fivethirtyeight')
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
# from sklearn.externals import joblib

# def SalvarModelo(my_model):
#     joblib.dump(my_model, "my_model.pkl") 
#     # and later... 
#     my_model_loaded = joblib.load("my_model.pkl")

def nashsutcliffe(evaluation, simulation):
    """
    Nash-Sutcliffe model efficinecy
        .. math::
         NSE = 1-\\frac{\\sum_{i=1}^{N}(e_{i}-s_{i})^2}{\\sum_{i=1}^{N}(e_{i}-\\bar{e})^2} 
    :evaluation: Observed data to compared with simulation data.
    :type: list
    :simulation: simulation data to compared with evaluation data
    :type: list
    :return: Nash-Sutcliff model efficiency
    :rtype: float
    """
    if len(evaluation) == len(simulation):
        s, e = np.array(simulation), np.array(evaluation)
        # s,e=simulation,evaluation
        mean_observed = np.nanmean(e)
        # compute numerator and denominator
        numerator = np.nansum((e - s) ** 2)
        denominator = np.nansum((e - mean_observed)**2)
        # compute coefficient
        return 1 - (numerator / denominator)

    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan

def preparacaodados(lags):
    dateparse=lambda dates: pd.datetime.strptime(dates, '%Y-%m')
    dados = pd.read_csv('Matriz_vazao_regress.csv', sep=';', parse_dates=['Month'], index_col='Month', date_parser =dateparse)

    
    dados_totais = dados.iloc[:, 15:16]  
    #Assimetria e Curtose
    dados_totais.skew()
    dados_totais.kurtosis()
    
    train = dados_totais[:312]
    test = dados_totais[312:]
    
    decomposition = seasonal_decompose(dados_totais.iloc[:,0:1], freq=12, model='additive')
    plt.rcParams['figure.figsize'] = 12, 5
    decomposition.plot()
    plt.show()
    
    
    dados['year'] = dados.index.year
    dados['month'] = dados.index.month
    df_pivot = pd.pivot_table(dados, values='Vazão', index='month', columns='year', aggfunc='mean')
    df_pivot.plot(figsize=(12,8))
    plt.legend().remove()
    plt.xlabel('Mês')
    plt.ylabel('Vazão')
    plt.show()
    dados.drop(['year', 'month'], axis=1, inplace=True)
    
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14,6), sharex=False, sharey=False)
    ax1 = plot_acf(dados_totais, lags=50, ax=ax1)
    ax2 = plot_pacf(dados_totais, lags=50, ax=ax2)
    plt.show()
    
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False, figsize=(12,5))
    dados.hist(ax=ax1)
    dados.plot(kind='kde', ax=ax2, figsize=(20,15))
    plt.show();
    
        #Determing rolling statistics
    rolmean = pd.Series(dados['Vazão']).rolling(window=12).mean()
    rolstd = pd.Series(dados['Vazão']).rolling(window=12).std()
    
    #Plot rolling statistics:
    orig = plt.plot(dados['Vazão'], color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Média Móvel')
    std = plt.plot(rolstd, color='black', label = 'Desvio Padrão Móvel')
    plt.legend(loc='best')
    plt.title('Média Móvel e Desvio Padrão (m³/s)')
    plt.show(block=False)
    
    print ('Resultadoss do Teste de Dickey-Fuller:')
    dftest = adfuller(dados['Vazão'])
    
    dfoutput = pd.Series(dftest[0:4], index=['Teste Estatístico','p-value','#Lags usados','Número de observações utilizado'])
    for key, value in dftest[4].items():
        dfoutput['Valor Crítico (%s)'%key] = value
    print(dfoutput)
    
    
    import pmdarima as pm
    model = pm.auto_arima(dados['Vazão'], d=1, D=1,
                      seasonal=True, m=12, trend='c', 
                      start_p=0, start_q=0, max_order=6, test='adf', stepwise=True, trace=True)
    model.summary()
    
    
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    model = SARIMAX(train.iloc[:,0],
                    order=(1,1,0),seasonal_order=(0,1,1,12))
    results = model.fit()
    results.summary()
        
    
    results.plot_diagnostics(figsize=(16, 8))
    plt.savefig('modeldiagnostics')
    plt.show()
    
        
    forecast_object = results.get_forecast(steps=len(test))
    
    mean = forecast_object.predicted_mean
    
    conf_int = forecast_object.conf_int()
    
    dates = test.index
    
    plt.figure(figsize=(16,8))

    # Plot past CO2 levels
    plt.plot(dados.index, dados.iloc[:,15:16], label='observado')
    
    # Plot the prediction means as line
    plt.plot(test.index, mean, label='previsto')
    
    # Shade between the confidence intervals
    plt.fill_between(dates, conf_int.iloc[:,0], conf_int.iloc[:,1],
    alpha=0.2)
    
    # Plot legend and show figure
    plt.legend()
    plt.savefig('predtest')
    plt.show()
    
    start=len(train)
    end=len(train)+len(test)-1
    predictions = results.predict(start=start, end=end, dynamic=False, typ='levels').rename('SARIMA(1,1,0)(0,1,1,12) Previsões')
    
    
    
    
    # Compare predictions to expected values
    for i in range(len(predictions)):
        print(f"predicted={predictions[i]:<11.10}, expected={test['Vazão'][i]}")
    
    
    # Plot predictions against known values
    title = 'Vazão Mensal Camargos (m³/s)'
    ax = test.iloc[:,0].plot(legend=True,figsize=(12,6),title=title)
    predictions.plot(legend=True)
    ax.autoscale(axis='x',tight=True)
    #ax.set(xlabel=xlabel, ylabel=ylabel);
    
    print(r2_score(test.iloc[:,0], predictions))
    
    
    evaluation_results = pd.DataFrame({'r2_score': r2_score(test.iloc[:,0], predictions)}, index=[0])
    evaluation_results['mean_absolute_error'] = mean_absolute_error(test.iloc[:,0], predictions)
    evaluation_results['mean_squared_error'] = mean_squared_error(test.iloc[:,0], predictions)
    evaluation_results['mean_absolute_percentage_error'] = np.mean(np.abs(predictions - test.iloc[:,0])/np.abs(test.iloc[:,0]))*100 
    
    evaluation_results
    
        
    pred_f = results.get_forecast(steps=60)
    pred_ci = pred_f.conf_int()
    ax = dados_totais.plot(label='observado', figsize=(14, 7))
    pred_f.predicted_mean.plot(ax=ax, label='previsto')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Data')
    ax.set_ylabel('Vazão Mensal Camargos')
    plt.legend()
    plt.show()
    # plt.plot(y_test, label='Dados Originais de Vazão | y ou t+1', color ='blue')    
    # plt.plot(X_test, label='Dados Passados | X ou t', color='orange')
    # plt.legend(loc='upper left')
    # plt.title('Dados passados em um período')
    # plt.show()
    
    print (nashsutcliffe(test.iloc[:,0], predictions))
    return X_train, y_train,X_test,y_test, X_total,y_total, scaler, X_tempo, y_tempo, X_anos, y_anos, X_meses, y_meses


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
    
def RodarSVM(X_train, y_train, X_test, y_test, scaler, lags, X_tempo, y_tempo, X_anos, y_anos, X_meses, y_meses, columns):
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
    
    print("Gerando Gráficos")    

        
    
    

    X_testmapa=X_test[:,0]
    y_predictmapa= y_predict[:,0]
    plt.rcParams["figure.figsize"] = (10,10)
    plt.plot(X_testmapa,X_testmapa, color='black')
    plt.scatter(X_testmapa,y_predictmapa, color='orange')     
    plt.legend(loc='best')
    plt.title('Resultados SVM Lag ' + str(lags))
    plt.ylabel('Previsto')
    plt.xlabel('Observado')
    plt.savefig('gráfico_svm_scatter_lag'+str(lags)+'.png')
    plt.show()
    



    df_y_predict=pd.DataFrame(y_predict)
    
    y_tempo = y_tempo.reshape(-1,1)
    df_y_tempo = pd.DataFrame(y_tempo, dtype='datetime64[D]')
    df_y_tempo.columns=['Datas']
    y_predict=df_y_predict.set_index(df_y_tempo.iloc[:,0],)
    y_predict.columns=['Vazão']
    
    y_test=y_test.reshape(-1,1)
    df_y_test=pd.DataFrame(y_test)
    
    y_tempo = y_tempo.reshape(-1,1)
    df_y_tempo = pd.DataFrame(y_tempo, dtype='datetime64[D]')
    df_y_tempo.columns=['Datas']
    y_test=df_y_test.set_index(df_y_tempo.iloc[:,0],)
    y_test.columns=['Vazão']
    
    X_test=X_test.reshape(-1,1)
    df_x_test=pd.DataFrame(X_test)
    
    X_tempo = X_tempo.reshape(-1,1)
    df_x_tempo = pd.DataFrame(X_tempo, dtype='datetime64[D]')
    
    X_test=df_x_test.set_index(df_x_tempo.iloc[:,0])
    X_test.columns=['Vazão']
    
    plt.rcParams["figure.figsize"] = (30,10)            
    plt.plot(X_test.iloc[:,0], label='Observado Lag ' + str(lags), color='orange')
    plt.plot(y_predict,label='Previsto', color='black')
    plt.legend(loc='best')
    plt.title('Resultados SVM')
    plt.xlabel('Data', fontsize=14)
    plt.ylabel('Vazão', fontsize=14)    
    plt.savefig(('gráfico_svm_lag'+str(lags)+'.png'))
    plt.show()
    

    
    return r2, mse, mae

        

    
def OLS(X_train, y_train, X_test, y_test, scaler, lags, X_tempo, y_tempo):
    
    reg = linear_model.LinearRegression()
    reg.fit(X_train, y_train)
    y_predict = reg.predict(X_test)
    X_test = scaler.inverse_transform(X_test)   
    
    y_test = scaler.inverse_transform(y_test)
    y_predict = scaler.inverse_transform(reg_predict)
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
    
    print("Gerando Gráficos")    

        
    
    

    X_testmapa=X_test[:,0]
    y_predictmapa= y_predict[:,0]
    plt.rcParams["figure.figsize"] = (10,10)
    plt.plot(X_testmapa,X_testmapa, color='black')
    plt.scatter(X_testmapa,y_predictmapa, color='orange')     
    plt.legend(loc='best')
    plt.title('Resultados SVM Lag ' + str(lags))
    plt.ylabel('Previsto')
    plt.xlabel('Observado')
    plt.savefig('gráfico_svm_scatter_lag'+str(lags)+'.png')
    plt.show()
    



    df_y_predict=pd.DataFrame(y_predict)
    
    y_tempo = y_tempo.reshape(-1,1)
    df_y_tempo = pd.DataFrame(y_tempo, dtype='datetime64[D]')
    df_y_tempo.columns=['Datas']
    y_predict=df_y_predict.set_index(df_y_tempo.iloc[:,0],)
    y_predict.columns=['Vazão']
    
    y_test=y_test.reshape(-1,1)
    df_y_test=pd.DataFrame(y_test)
    
    y_tempo = y_tempo.reshape(-1,1)
    df_y_tempo = pd.DataFrame(y_tempo, dtype='datetime64[D]')
    df_y_tempo.columns=['Datas']
    y_test=df_y_test.set_index(df_y_tempo.iloc[:,0],)
    y_test.columns=['Vazão']
    
    X_test=X_test.reshape(-1,1)
    df_x_test=pd.DataFrame(X_test)
    
    X_tempo = X_tempo.reshape(-1,1)
    df_x_tempo = pd.DataFrame(X_tempo, dtype='datetime64[D]')
    
    X_test=df_x_test.set_index(df_x_tempo.iloc[:,0])
    X_test.columns=['Vazão']
    
    plt.rcParams["figure.figsize"] = (30,10)            
    plt.plot(X_test.iloc[:,0], label='Observado Lag ' + str(lags), color='orange')
    plt.plot(y_predict,label='Previsto', color='black')
    plt.legend(loc='best')
    plt.title('Resultados OLS')
    plt.xlabel('Data', fontsize=14)
    plt.ylabel('Vazão', fontsize=14)    
    plt.savefig(('gráfico_ols_lag'+str(lags)+'.png'))
    plt.show()
    
    
    
    return r2, mse,mae

def RodarMLP(X_train, y_train, X_test, y_test, scaler, lags,  X_tempo, y_tempo, X_anos, y_anos, X_meses, y_meses):
    print("Rodando Modelo")
    mlp = MLPRegressor()
    model = MLPRegressor(activation='relu', alpha=0.001, batch_size=50, hidden_layer_sizes=(8,9,2), max_iter=1000, solver='adam')
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    # model = KerasRegressor(build_fn=CriarMLP(lags), epochs=1000, batch_size=10, verbose=1)
    # hidden_layer = GerarHiddenLayers()
    # parameter_space = {
    # 'hidden_layer_sizes': hidden_layer,
    # 'activation': ['tanh', 'relu','softplus'],
    # 'solver': ['sgd', 'adam'],
    # 'alpha': [0.0001, 0.05,0.1, 0.01],
    # 'batch_size':[50],
    # 'max_iter':[1000]    
    # }
    
    # model = GridSearchCV(mlp, parameter_space, n_jobs=6, cv=3, verbose=1, scoring=scorer)
    print("Alinhando Modelo")
    model.fit(X_train, y_train)
    print("Prevendo para dados de teste")
    y_predict = model.predict(X_test)
    
    y_test = scaler.inverse_transform(y_test)
    y_predict = scaler.inverse_transform(y_predict)
    X_test = scaler.inverse_transform(X_test)   
    # calculate Pearson's correlation
    
    print("Calculando Pearson")
    pearson = pearsonr(y_test, y_predict)
    print("pearson:" + str(pearson))
    r2 = r2_score(y_test, y_predict)
    print("r2:" + str(r2))
    mse = mean_squared_error(y_test, y_predict)
    mae = mean_absolute_error(y_test, y_predict)
    # Best paramete set
    # print('Best parameters found:\n', model.best_params_)
    # GravaremTXT("Melhores Parâmetros: " + str(model.best_params_))
    # All results
    # means = model.cv_results_['mean_test_score']
    # stds = model.cv_results_['std_test_score']
    # for mean, std, params in zip(means, stds, model.cv_results_['params']):
        # print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        # GravaremTXT("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        
       # alteração de variável para criação de gráficos, adicionando data
    y_predict=y_predict.reshape(-1,1)
    
    print("Gerando Gráficos")    
    X_testmapa=X_test[:,0]
    y_predictmapa= y_predict[:,0]
    plt.rcParams["figure.figsize"] = (10,10)
    plt.plot(X_testmapa,X_testmapa, color='black')
    plt.scatter(X_testmapa,y_predictmapa, color='orange')     
    plt.legend(loc='best')
    plt.title('Resultados MLP Lag ' + str(lags))
    plt.ylabel('Previsto')
    plt.xlabel('Observado')
    plt.savefig('gráfico_mlp_scatter_lag'+str(lags)+'.png')
    plt.show()
    



    df_y_predict=pd.DataFrame(y_predict)
    
    y_tempo = y_tempo.reshape(-1,1)
    df_y_tempo = pd.DataFrame(y_tempo, dtype='datetime64[D]')
    df_y_tempo.columns=['Datas']
    y_predict=df_y_predict.set_index(df_y_tempo.iloc[:,0],)
    y_predict.columns=['Vazão']
    
    y_test=y_test.reshape(-1,1)
    df_y_test=pd.DataFrame(y_test)
    
    # y_tempo = y_tempo.reshape(-1,1)
    df_y_tempo = pd.DataFrame(y_tempo, dtype='datetime64[D]')
    df_y_tempo.columns=['Datas']
    df_y_tempo=df_y_tempo.sort_values(by="Datas")
    
    y_test=df_y_test.set_index(df_y_tempo.iloc[:,0],)
    y_test.columns=['Vazão']
    
    X_test=X_test.reshape(-1,1)
    df_x_test=pd.DataFrame(X_test)
    
    X_tempo = X_tempo.reshape(-1,1)
    df_x_tempo = pd.DataFrame(X_tempo, dtype='datetime64[D]')
    Xtesttest = pd.DataFrame(X_test[:,0:1])
    
    X_test=X_test.set_index(df_y_tempo.iloc[:,0])
    X_test.columns=['Vazão']
    
    plt.rcParams["figure.figsize"] = (30,10)            
    plt.plot(X_test.iloc[:,0], label='Observado Lag ' + str(lags), color='orange')
    plt.plot(y_predict,label='Previsto', color='black')
    plt.legend(loc='best')
    plt.title('Resultados MLP')
    plt.xlabel('Data', fontsize=14)
    plt.ylabel('Vazão', fontsize=14)    
    plt.savefig(('gráfico_mlp_lag'+str(lags)+'.png'))
    plt.show()
    
    return r2, mse,mae
def GerarHiddenLayers():

    layers = []
    X=np.arange(1,10,1).tolist()
    Y=np.arange(1,10,1).tolist()
    Z=np.arange(1,10,1).tolist()
    
    for j in X:
        for k in Y:
            for l in Z:              
                layer = tuple((j,k,l))
                
                layers.append(layer)
    return layers              
                    
def CriarMLP(lags):
    #create model  
    model = Sequential()
    model.add(Dense(32,input_dim=lags, activation='tanh'))
    model.add(Dense(16, input_dim=lags, activation='tanh'))
    model.add(Dense(16, input_dim=lags, activation='tanh'))
    model.add(Dense(1, input_dim=lags))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def RodarModelos():
    
    t0 = time.time()
    # r2=[0.519414749386341,0.2938693108439203,0.43726183758534454,0.4743342229830483,0.33934437066068057,0.4184729169398904,0.3277643323055064,0.2985764009381916, 0.2753468193319707, 0.23508359585957805,0.2398079428261859,0.19006508466344663]
    maelist=[]
    r2list=[]
    mselist=[]
    maelistols=[]
    r2listols=[]
    mselistols=[]
    maelistmlp=[]
    r2listmlp=[]
    mselistmlp=[]
    columns=['Meses', 'Vazão']
    
    for lags in range(1, 13):
        lags=1
        print("Lag " + str(lags))
        X_train, y_train, X_test, y_test, X_total,y_total, scaler, X_tempo, y_tempo, X_anos,y_anos,X_meses,y_meses = preparacaodados(lags)
        # RodarSVMGridsearch(X_train, y_train, X_test, y_test,X_total,y_total, scaler)
        # print("SVM")
        # columns.append("Vazão")
        r2, mse, mae =RodarSVM(X_train, y_train, X_test, y_test, scaler, lags, X_tempo, y_tempo, X_anos,y_anos,X_meses,y_meses, columns)
        mselist.append(mse)
        r2list.append(r2)
        maelist.append(mae)        
        
        # X_train, y_train, X_test, y_test, X_total,y_total, scaler, X_tempo, y_tempo, X_anos,y_anos,X_meses,y_meses = preparacaodados(lags)
        
        # r2mlp, msemlp, maemlp=RodarMLP(X_train, y_train, X_test, y_test, scaler, lags,  X_tempo, y_tempo, X_anos,y_anos,X_meses,y_meses)
        # mselistmlp.append(msemlp)
        # r2listmlp.append(r2mlp)
        # maelistmlp.append(maemlp)
        # print("OLS")
        
        
        # r2ols, mseols, maeols= OLS(X_train, y_train, X_test, y_test, scaler,  X_tempo, y_tempo)
        # mselistols.append(mseols)
        # r2listols.append(r2ols)
        # maelistols.append(maeols)
        # t1 = time.time() -t0   
        # print("tempo decorrido:" + str(t1))
    # r2concat = np.stack((r2list,r2listols), axis=1)    
    
        
            
    x1 = np.arange(1,13)
    x2 = [x + 0.25 for x in x1]
    x3 = [x + 0.25 for x in x2]
    x4 = [x + 0.25 for x in x3]
    x5 = [x + 0.25 for x in x4]
    x6 = [x + 0.25 for x in x5]
    x7 = [x + 0.25 for x in x6]
    x8 = [x + 0.25 for x in x7]
    x9 = [x + 0.25 for x in x8]
    x10 = [x + 0.25 for x in x9]
    x11 = [x + 0.25 for x in x9]
    x12 = [x + 0.25 for x in x9]
    

    

    
    plt.title("R-Squared por Lag")
    plt.bar(x1, r2list, width=0.25, label = 'R2 SVM', color = 'orange')
    plt.bar(x2, r2listols, width=0.25, label = 'R2 OLS', color = 'firebrick')
    plt.bar(x3, r2listmlp, width=0.25, label = 'R2 MLP', color = 'red')    
    plt.legend(loc='best')    
    plt.xlabel('Lag', fontsize=14)       
    plt.xticks(np.arange(0,13,1))
    plt.savefig(('gráfico_svm_resumor2_lag.png'))
    plt.show()    
    
    plt.title("Mean Squared Error ")
    plt.bar(x1, mselist, width=0.25, label = 'MSE SVM', color = 'lawngreen')
    plt.bar(x2, mselistols, width=0.25, label = 'MSE OLS', color = 'darkolivegreen')
    plt.bar(x3, mselistmlp, width=0.25, label = 'MSE MLP', color = 'green')
    plt.legend(loc='best')    
    plt.xlabel('Lag', fontsize=14)    
    plt.xticks(np.arange(0,13,1))
    plt.savefig(('gráfico_svm_resumomse_lag.png'))
    plt.show()    
    
    plt.title("Mean Absolute Error")
    plt.bar(x1, maelist, width=0.25, label = 'MAE SVM', color = 'deepskyblue')
    plt.bar(x2, maelistols, width=0.25, label = 'MAE OLS', color = 'cornflowerblue')
    plt.bar(x3, maelistmlp, width=0.25, label = 'MAE MLP', color = 'mediumblue')
    plt.legend(loc='best')    
    plt.xlabel('Lag', fontsize=14)    
    plt.xticks(np.arange(0,13,1))
    plt.savefig(('gráfico_svm_resumomae_lag.png'))
    
    plt.show()    
    # t1 = time.time() - t0   
    # print("tempo decorrido total:" + str(t1))