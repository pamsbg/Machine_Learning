# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 21:43:49 2020

@author: pamsb
"""

#redes neurais artificiais
from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pylab as plt
#matplotlib.pylab inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=15,6
import pandas as pd
from scipy.stats import pearsonr
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, cross_val_predict, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer
from sklearn.preprocessing import MinMaxScaler
import math
import time


def preparacaoDados(lags):
    dateparse=lambda dates: pd.datetime.strptime(dates, '%Y-%m')
    df = pd.read_csv('Matriz_vazao_regress.csv', sep=';', parse_dates=['Month'], index_col='Month', date_parser =dateparse)
    
    dataprep=df.iloc[:,13:14]
    data=dataprep.values
    
    
    data = data.astype('float32')
    scaler = MinMaxScaler(feature_range=(0,1))
    data = scaler.fit_transform(data)
    splits = TimeSeriesSplit(n_splits=2)
    plt.figure(1)
    index = 1
    for train_index, test_index in splits.split(data):
    	train = data[train_index]
    	test = data[test_index]
    	print('Observations: %d' % (len(train) + len(test)))
    	print('Training Observations: %d' % (len(train)))
    	print('Testing Observations: %d' % (len(test)))
    	plt.subplot(310 + index)
    	plt.plot(train)
    	plt.plot([None for i in train] + [x for x in test])
    	index += 1
    plt.show()
    
    X_train,y_train = prepare_data(train,lags)
    X_test,y_test = prepare_data(test,lags)
    y_true = y_test
    
    plt.plot(y_test, label='Dados Originais de Vazão | y ou t+1', color ='blue')    
    plt.plot(X_test, label='Dados Passados | X ou t', color='orange')
    plt.legend(loc='upper left')
    plt.title('Dados passados em um período')
    plt.show()
   
    return X_train,y_train, X_test, y_test



def prepare_data(data, lags):
    X,y = [],[]
    for row in range(len(data)-lags-1):
        a = data[row:(row+lags),0]
        X.append(a)
        y.append(data[row-lags,0])
    return np.array(X),np.array(y)


 
    
    # X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1]))
    # X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1]))
    
def mycustomscorer(y_test, prediction):
    mycustomscorer, _ = pearsonr(y_test, prediction)
    return mycustomscorer
    

    
    
def CriarMLP():
    #create model  
    model = Sequential()
    model.add(Dense(1,input_dim=lags, activation='softplus'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='RMSprop')
    return model

# train_predict = model.predict(X_train)
# test_predict = model.predict(X_test)
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
                    
        
def GravaremTXT(output):
    file = open("saídaGridSearch.txt","w") 
    file.write(output) 
    file.close() 
        
    
    
def RodarMLP(X_train, y_train, X_test, y_test):
    print("Rodando Modelo")
    model = MLPRegressor(activation='tanh', alpha=0.05, batch_size=50, hidden_layer_sizes=(5,6,9), max_iter=1000, solver='adam')
    
    # model = KerasRegressor(build_fn=CriarMLP, epochs=1000, batch_size=10, verbose=0)
    # hidden_layer = GerarHiddenLayers()
    # parameter_space = {
    # 'hidden_layer_sizes': hidden_layer,
    # 'activation': ['tanh', 'relu'],
    # 'solver': ['sgd', 'adam'],
    # 'alpha': [0.0001, 0.05],
    # 'batch_size':[50],
    # 'max_iter':[1000]    
    # }
    
    # model = GridSearchCV(mlp, parameter_space, n_jobs=6, cv=3, verbose=1)
    print("Alinhando Modelo")
    model.fit(X_train, y_train)
    
    print("Prevendo para dados de teste")
    prediction = model.predict(X_test)
    # calculate Pearson's correlation
    
    mlp_r2_predict, _ = pearsonr(y_test, prediction)
    print(str(mlp_r2_predict))
    GravaremTXT("r2_predict" + str(mlp_r2_predict))
    mlp_mse_predict = mean_squared_error(y_test, prediction)
    mlp_mae_predict = mean_absolute_error(y_test, prediction)
    # Best paramete set
    # print('Best parameters found:\n', model.best_params_)
    # GravaremTXT("Melhores Parâmetros: " + str(model.best_params_))
    # # All results
    # means = model.cv_results_['mean_test_score']
    # stds = model.cv_results_['std_test_score']
    # for mean, std, params in zip(means, stds, model.cv_results_['params']):
    #     # print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    #     GravaremTXT("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        
def RodarModelos():
    
    t0 = time.time()
    
    for lags in range(1, 13):
        print("Iniciando Loop, Lag:" + str(lags))
        print("Lag " + str(lags))
        GravaremTXT("Lag" + str(lags))
        X_train, y_train, X_test, y_test = preparacaoDados(lags)
        RodarMLP(X_train, y_train, X_test, y_test)
        t1 = time.time() -t0   
        print("tempo decorrido:" + str(t1))
    t1 = time.time() -t0   
    print("tempo decorrido total:" + str(t1))
    

