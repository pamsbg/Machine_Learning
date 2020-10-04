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
# from sklearn.externals import joblib

# def SalvarModelo(my_model):
#     joblib.dump(my_model, "my_model.pkl") 
#     # and later... 
#     my_model_loaded = joblib.load("my_model.pkl")



def preparacaodados(lags):
    dateparse=lambda dates: pd.datetime.strptime(dates, '%Y-%m')
    dados = pd.read_csv('Matriz_vazao_regress.csv', sep=';', parse_dates=['Month'], index_col='Month', date_parser =dateparse)
    
    dados_totais = dados.iloc[:, 15:16].values
    meses = dados.iloc[312:,1:2].values
    anos = dados.iloc[312:,0:1].values
    meses=meses.reshape(-1,1)
    anos=anos.reshape(-1,1)
    tempo = np.array(dados.index[312:].tolist(),dtype='datetime64[D]')
    tempo=tempo.reshape(-1,1)
    X_anos ,y_anos = prepare_data(anos,lags)
    X_meses ,y_meses = prepare_data(meses,lags)
    X_tempo ,y_tempo = prepare_data(tempo,lags)
    scaler = preprocessing.StandardScaler()
    dados_escalados = scaler.fit_transform(dados_totais)
    treino = dados.iloc[:312, 15:16].values
    treino_escalados = scaler.fit_transform(treino)
    teste = dados.iloc[312:432, 15:16].values
    teste_escalados = scaler.fit_transform(teste)
    validacao = dados.iloc[346:432, 15:16]

    
    X_total,y_total = prepare_data(dados_escalados,lags)
    X_train,y_train = prepare_data(treino_escalados,lags)
    X_test,y_test = prepare_data(teste_escalados,lags)
    y_true = y_test
    
    plt.plot(y_test, label='Dados Originais de Vazão | y ou t+1', color ='blue')    
    plt.plot(X_test, label='Dados Passados | X ou t', color='orange')
    plt.legend(loc='upper left')
    plt.title('Dados passados em um período')
    plt.show()
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
    df_y_predict=pd.DataFrame(y_predict)
    
    y_tempo = y_tempo.reshape(-1,1)
    df_y_tempo = pd.DataFrame(y_tempo, dtype='datetime64[D]')
    df_y_tempo.columns=['Datas']
    y_predict=df_y_predict.set_index(df_y_tempo.iloc[:,0],)
    y_predict.columns=['Vazao']
    
    y_test=y_test.reshape(-1,1)
    df_y_test=pd.DataFrame(y_test)
    
    y_tempo = y_tempo.reshape(-1,1)
    df_y_tempo = pd.DataFrame(y_tempo, dtype='datetime64[D]')
    df_y_tempo.columns=['Datas']
    y_test=df_y_test.set_index(df_y_tempo.iloc[:,0],)
    y_test.columns=['Vazao']
    
    X_test=X_test.reshape(-1,1)
    df_x_test=pd.DataFrame(X_test)
    
    X_tempo = X_tempo.reshape(-1,1)
    df_x_tempo = pd.DataFrame(X_tempo, dtype='datetime64[D]')
    
    X_test=df_x_test.set_index(df_x_tempo.iloc[:,0])
    X_test.columns=['Vazao']
    
    y_meses = y_meses.reshape(-1,1)
    df_meses=pd.DataFrame(y_meses)
    df_meses.columns=['Meses']
    columns_def=np.array(columns)
    frameobservado=[df_meses,y_test]
    filtermesesobservado=pd.DataFrame(np.hstack((frameobservado)), columns=columns_def)
    
    obsfilterjaneiro =filtermesesobservado.loc[filtermesesobservado['Meses']==1]
    obsfilterfevereiro =filtermesesobservado.loc[filtermesesobservado['Meses']==2]
    obsfiltermarco =filtermesesobservado.loc[filtermesesobservado['Meses']==3]
    obsfilterabril =filtermesesobservado.loc[filtermesesobservado['Meses']==4]
    obsfiltermaio =filtermesesobservado.loc[filtermesesobservado['Meses']==5]
    obsfilterjunho =filtermesesobservado.loc[filtermesesobservado['Meses']==6]
    obsfilterjulho =filtermesesobservado.loc[filtermesesobservado['Meses']==7]
    obsfilteragosto =filtermesesobservado.loc[filtermesesobservado['Meses']==8]
    obsfiltersetembro =filtermesesobservado.loc[filtermesesobservado['Meses']==9]
    obsfilteroutubro =filtermesesobservado.loc[filtermesesobservado['Meses']==10]
    obsfilternovembro =filtermesesobservado.loc[filtermesesobservado['Meses']==11]
    obsfilterdezembro =filtermesesobservado.loc[filtermesesobservado['Meses']==12]
    
    frame=[df_meses,y_predict]
    filtermeses=pd.DataFrame(np.hstack((frame)), columns=columns_def)
    filterjaneiro =filtermeses.loc[filtermeses['Meses']==1]
    filterfevereiro =filtermeses.loc[filtermeses['Meses']==2]
    filtermarco =filtermeses.loc[filtermeses['Meses']==3]
    filterabril =filtermeses.loc[filtermeses['Meses']==4]
    filtermaio =filtermeses.loc[filtermeses['Meses']==5]
    filterjunho =filtermeses.loc[filtermeses['Meses']==6]
    filterjulho =filtermeses.loc[filtermeses['Meses']==7]
    filteragosto =filtermeses.loc[filtermeses['Meses']==8]
    filtersetembro =filtermeses.loc[filtermeses['Meses']==9]
    filteroutubro =filtermeses.loc[filtermeses['Meses']==10]
    filternovembro =filtermeses.loc[filtermeses['Meses']==11]
    filterdezembro =filtermeses.loc[filtermeses['Meses']==12]
    
    
     
    
    r2jan = r2_score(obsfilterjaneiro['Vazao'], filterjaneiro['Vazao'])
    r2fev = r2_score(obsfilterfevereiro['Vazao'], filterfevereiro['Vazao'])
    r2mar = r2_score(obsfiltermarco['Vazao'], filtermarco['Vazao'])
    r2abr = r2_score(obsfilterabril['Vazao'], filterabril['Vazao'])
    r2mai = r2_score(obsfiltermaio['Vazao'], filtermaio['Vazao'])
    r2jun = r2_score(obsfilterjunho['Vazao'], filterjunho['Vazao'])
    r2jul = r2_score(obsfilterjulho['Vazao'], filterjulho['Vazao'])
    r2ago = r2_score(obsfilteragosto['Vazao'], filteragosto['Vazao'])
    r2set = r2_score(obsfiltersetembro['Vazao'], filtersetembro['Vazao'])
    r2out = r2_score(obsfilteroutubro['Vazao'], filteroutubro['Vazao'])
    r2nov = r2_score(obsfilternovembro['Vazao'], filternovembro['Vazao'])
    r2dez = r2_score(obsfilterdezembro['Vazao'], filterdezembro['Vazao'])
    
    print("r2:" + str(r2))
    mse= mean_squared_error(y_test,y_predict)
    msejan = mean_squared_error(obsfilterjaneiro['Vazao'], filterjaneiro['Vazao'])
    msefev = mean_squared_error(obsfilterfevereiro['Vazao'], filterfevereiro['Vazao'])
    msemar = mean_squared_error(obsfiltermarco['Vazao'], filtermarco['Vazao'])
    mseabr = mean_squared_error(obsfilterabril['Vazao'], filterabril['Vazao'])
    msemai = mean_squared_error(obsfiltermaio['Vazao'], filtermaio['Vazao'])
    msejun = mean_squared_error(obsfilterjunho['Vazao'], filterjunho['Vazao'])
    msejul = mean_squared_error(obsfilterjulho['Vazao'], filterjulho['Vazao'])
    mseago = mean_squared_error(obsfilteragosto['Vazao'], filteragosto['Vazao'])
    mseset = mean_squared_error(obsfiltersetembro['Vazao'], filtersetembro['Vazao'])
    mseout = mean_squared_error(obsfilteroutubro['Vazao'], filteroutubro['Vazao'])
    msenov = mean_squared_error(obsfilternovembro['Vazao'], filternovembro['Vazao'])
    msedez = mean_squared_error(obsfilterdezembro['Vazao'], filterdezembro['Vazao'])
    print("mse:" + str(mse))
    mae= mean_absolute_error(y_test,y_predict)
    maejan = mean_absolute_error(obsfilterjaneiro['Vazao'], filterjaneiro['Vazao'])
    maefev = mean_absolute_error(obsfilterfevereiro['Vazao'], filterfevereiro['Vazao'])
    maemar = mean_absolute_error(obsfiltermarco['Vazao'], filtermarco['Vazao'])
    maeabr = mean_absolute_error(obsfilterabril['Vazao'], filterabril['Vazao'])
    maemai = mean_absolute_error(obsfiltermaio['Vazao'], filtermaio['Vazao'])
    maejun = mean_absolute_error(obsfilterjunho['Vazao'], filterjunho['Vazao'])
    maejul = mean_absolute_error(obsfilterjulho['Vazao'], filterjulho['Vazao'])
    maeago = mean_absolute_error(obsfilteragosto['Vazao'], filteragosto['Vazao'])
    maeset = mean_absolute_error(obsfiltersetembro['Vazao'], filtersetembro['Vazao'])
    maeout = mean_absolute_error(obsfilteroutubro['Vazao'], filteroutubro['Vazao'])
    maenov = mean_absolute_error(obsfilternovembro['Vazao'], filternovembro['Vazao'])
    maedez = mean_absolute_error(obsfilterdezembro['Vazao'], filterdezembro['Vazao'])
    print("mae:" + str(mae))
    
    
    
    
  
    
    
    
    print("Gerando Gráficos")    
    plt.plot(X_test, label='Observado Lag ' + str(lags), color='orange')
    plt.plot(y_predict,label='Previsto', color='black')
    plt.legend(loc='best')
    plt.title('Resultados')
    plt.xlabel('Data', fontsize=14)
    plt.ylabel('Vazão', fontsize=14)    
    plt.savefig(('gráfico_svm_lag'+str(lags)+'.png'))
    plt.show()
    return r2, mse, mae, r2jan ,    r2fev ,    r2mar ,    r2abr ,    r2mai ,    r2jun ,    r2jul ,    r2ago ,    r2set ,    r2out ,    r2nov ,    r2dez, msejan ,    msefev ,    msemar ,    mseabr ,    msemai ,    msejun ,    msejul ,    mseago ,    mseset ,    mseout ,    msenov ,    msedez, maejan ,    maefev ,    maemar ,    maeabr ,    maemai ,    maejun ,    maejul ,    maeago ,    maeset ,    maeout ,    maenov ,    maedez

        

    
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

def RodarMLP(X_train, y_train, X_test, y_test, scaler, lags):
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
    r2listanualporlag=[]
    r2listjan=[]
    r2listfev=[]
    r2listmar=[]
    r2listabr=[]
    r2listmai=[]
    r2listjun=[]
    r2listjul=[]
    r2listago=[]
    r2listset=[]
    r2listout=[]
    r2listnov=[]
    r2listdez=[]
    mselistjan=[]
    mselistfev=[]
    mselistmar=[]
    mselistabr=[]
    mselistmai=[]
    mselistjun=[]
    mselistjul=[]
    mselistago=[]
    mselistset=[]
    mselistout=[]
    mselistnov=[]
    mselistdez=[]
    maelistjan=[]
    maelistfev=[]
    maelistmar=[]
    maelistabr=[]
    maelistmai=[]
    maelistjun=[]
    maelistjul=[]
    maelistago=[]
    maelistset=[]
    maelistout=[]
    maelistnov=[]
    maelistdez=[]
    columns=['Meses', 'Vazao']
    for lags in range(1, 13):
        print("Lag " + str(lags))
        X_train, y_train, X_test, y_test, X_total,y_total, scaler, X_tempo, y_tempo, X_anos,y_anos,X_meses,y_meses = preparacaodados(lags)
        # RodarSVMGridsearch(X_train, y_train, X_test, y_test,X_total,y_total, scaler)
        # print("SVM")
        # columns.append("Vazao")
        r2, mse, mae, r2jan ,    r2fev ,    r2mar ,    r2abr ,    r2mai ,    r2jun ,    r2jul ,    r2ago ,    r2set ,    r2out ,    r2nov ,    r2dez, msejan ,    msefev ,    msemar ,    mseabr ,    msemai ,    msejun ,    msejul ,    mseago ,    mseset ,    mseout ,    msenov ,    msedez,maejan ,    maefev ,    maemar ,    maeabr ,    maemai ,    maejun ,    maejul ,    maeago ,    maeset ,    maeout ,    maenov ,    maedez =RodarSVM(X_train, y_train, X_test, y_test, scaler, lags, X_tempo, y_tempo, X_anos,y_anos,X_meses,y_meses, columns)
        mselist.append(mse)
        r2list.append(r2)
        maelist.append(mae)
        r2listanualporlag.append(r2jan)
        r2listanualporlag.append(r2fev)
        r2listanualporlag.append(r2mar)
        r2listanualporlag.append(r2abr)
        r2listanualporlag.append(r2mai)
        r2listanualporlag.append(r2jun)
        r2listanualporlag.append(r2jul)
        r2listanualporlag.append(r2ago)
        r2listanualporlag.append(r2set)
        r2listanualporlag.append(r2out)
        r2listanualporlag.append(r2nov)
        r2listanualporlag.append(r2dez)
        r2listjan.append(r2jan)
        r2listfev.append(r2fev)
        r2listmar.append(r2mar)
        r2listabr.append(r2abr)
        r2listmai.append(r2mai)
        r2listjun.append(r2jun)
        r2listjul.append(r2jul)
        r2listago.append(r2ago)
        r2listset.append(r2set)
        r2listout.append(r2out)
        r2listnov.append(r2nov)
        r2listdez.append(r2dez)
        mselistjan.append(msejan)
        mselistfev.append(msefev)
        mselistmar.append(msemar)
        mselistabr.append(mseabr)
        mselistmai.append(msemai)
        mselistjun.append(msejun)
        mselistjul.append(msejul)
        mselistago.append(mseago)
        mselistset.append(mseset)
        mselistout.append(mseout)
        mselistnov.append(msenov)
        mselistdez.append(msedez)
        maelistjan.append(maejan)
        maelistfev.append(maefev)
        maelistmar.append(maemar)
        maelistabr.append(maeabr)
        maelistmai.append(maemai)
        maelistjun.append(maejun)
        maelistjul.append(maejul)
        maelistago.append(maeago)
        maelistset.append(maeset)
        maelistout.append(maeout)
        maelistnov.append(maenov)
        maelistdez.append(maedez)
        
        r2mlp, msemlp, maemlp=RodarMLP(X_train, y_train, X_test, y_test, scaler, lags)
        mselistmlp.append(msemlp)
        r2listmlp.append(r2mlp)
        maelistmlp.append(maemlp)
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
    x3 = [x + 0.25 for x in x2]
    
    plt.title("R-Squared por mês")
    plt.bar(x1, r2listanualporlag, width=0.25, label = 'R2 SVM', color = 'orange')   
    plt.legend(loc='best')    
    plt.xlabel('Mês', fontsize=14)    
    plt.savefig(('gráfico_svm_resumor2_lag.png'))
    plt.show()    
    
    plt.title("R-Squared por Lag")
    plt.bar(x1, r2list, width=0.25, label = 'R2 SVM', color = 'orange')
    plt.bar(x2, r2listols, width=0.25, label = 'R2 OLS', color = 'firebrick')
    plt.bar(x3, r2listmlp, width=0.25, label = 'R2 MLP', color = 'red')
    plt.legend(loc='best')    
    plt.xlabel('Lag', fontsize=14)    
    plt.savefig(('gráfico_svm_resumor2_lag.png'))
    plt.show()    
    
    plt.title("Mean Squared Error ")
    plt.bar(x1, mselist, width=0.25, label = 'MSE SVM', color = 'lawngreen')
    plt.bar(x2, mselistols, width=0.25, label = 'MSE OLS', color = 'darkolivegreen')
    plt.bar(x3, mselistmlp, width=0.25, label = 'MSE MLP', color = 'green')
    plt.legend(loc='best')    
    plt.xlabel('Lag', fontsize=14)    
    
    plt.savefig(('gráfico_svm_resumomse_lag.png'))
    plt.show()    
    
    plt.title("Mean Absolut Error")
    plt.bar(x1, maelist, width=0.25, label = 'MAE SVM', color = 'deepskyblue')
    plt.bar(x2, maelistols, width=0.25, label = 'MAE OLS', color = 'cornflowerblue')
    plt.bar(x3, maelistmlp, width=0.25, label = 'MAE MLP', color = 'mediumblue')
    plt.legend(loc='best')    
    plt.xlabel('Lag', fontsize=14)    
    plt.savefig(('gráfico_svm_resumomae_lag.png'))
    plt.show()    
    # t1 = time.time() - t0   
    # print("tempo decorrido total:" + str(t1))