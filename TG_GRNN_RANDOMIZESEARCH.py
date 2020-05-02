# -*- coding: utf-8 -*-
"""
Created on Sat May  2 20:10:36 2020

@author: pamsb
"""


# -*- coding: utf-8 -*-
"""
Created on Sat May  2 18:35:15 2020

@author: pamsb
"""


# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 21:43:49 2020

@author: pamsb
"""




#redes neurais artificiais
from neupy import algorithms
import heapq
import numpy as np
import matplotlib.pylab as plt
#matplotlib.pylab inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=15,6
import pandas as pd
from scipy.stats import pearsonr
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, cross_val_predict, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer
from sklearn.preprocessing import MinMaxScaler
import math
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


resultados_grnn = np.zeros((13,13))

resultados_rl = np.zeros((13,13))






for lags in range(1,13):

    print("Iniciando Loop, Lag:" + str(lags))
    def prepare_data(data, lags):
        X,y = [],[]
        for row in range(len(data)-lags-1):
            a = data[row:(row+lags),0]
            X.append(a)
            y.append(data[row-lags,0])
        return np.array(X),np.array(y)
                      
    
        
    # lags =12
    X_train,y_train = prepare_data(train,lags)
    X_test,y_test = prepare_data(test,lags)
    y_true = y_test
    
    plt.plot(y_test, label='Dados Originais de Vazão | y ou t+1', color ='blue')    
    plt.plot(X_test, label='Dados Passados | X ou t', color='orange')
    plt.legend(loc='upper left')
    plt.title('Dados passados em um período')
    plt.show()
    
    X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1]))
    
    def mycustomscorer(y_test, prediction):
        mycustomscorer, _ = pearsonr(y_test, prediction)
        return mycustomscorer
    
    print("Criando função especial para cáluclo de pearson")
    my_scorer = make_scorer(mycustomscorer, greater_is_better=True)
    
   
    
    # train_predict = model.predict(X_train)
    # test_predict = model.predict(X_test)
    print("Rodando Modelo")
        
    grnn = algorithms.GRNN(std=0.1)
    grnn.train(X_train, y_train)
    
    
    
    
       
    print("Criando crossval de resultados")
    # grnn_r2_train = cross_val_score(grnn, X_train, y_train, cv=splits, scoring=my_scorer)
    # grnn_r2_test = cross_val_score(grnn, X_test, y_test, cv=splits, scoring=my_scorer)
    # grnn_mse_train = cross_val_score(grnn, X_train, y_train, cv=splits, scoring='neg_mean_squared_error')
    # grnn_mae_train = cross_val_score(grnn, X_train, y_train, cv=splits, scoring='neg_mean_absolute_error')
    # grnn_mse_test = cross_val_score(grnn, X_test, y_test, cv=splits, scoring='neg_mean_squared_error')
    # grnn_mae_test = cross_val_score(grnn, X_test, y_test, cv=splits, scoring='neg_mean_absolute_error')
    # # mlp_r2=r2_score(y_test,test_predict)

    print("Alinhando Modelo")
    grnn.fit(X_train, y_train)
    
    print("Prevendo para dados de teste")
    prediction_train = grnn.predict(X_train).reshape(X_train.shape[0],)
    prediction = grnn.predict(X_test).reshape(X_test.shape[0],)
    # calculate Pearson's correlation
    
    grnn_r2_test, _= np.array(pearsonr(y_test, prediction))
    grnn_r2_train = np.array(pearsonr(y_train, prediction_train))
    
    
    grnn_mse_train = mean_squared_error(y_train, prediction_train)
    grnn_mae_train = mean_absolute_error(y_train, prediction_train)
    
    grnn_mse_test = mean_squared_error(y_test, prediction)
    grnn_mae_test = mean_absolute_error(y_test, prediction)
    def rmsle(expected, predicted):
        log_expected = np.log1p(expected + 1)
        log_predicted = np.log1p(predicted + 1)
        squared_log_error = np.square(log_expected - log_predicted)
        return np.sqrt(np.mean(squared_log_error))


    def scorer(network, X, y):
        result = network.predict(X)
        return rmsle(result, y)
    
    def report(results, n_top=3):
        ranks = heapq.nlargest(n_top, results['rank_test_score'])

        for i in ranks:
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results['mean_test_score'][candidate],
                    results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")


    print("Run Random Search CV")
    
    random_search = RandomizedSearchCV(
        algorithms.GRNN(std=0.1, verbose=False),
        param_distributions={'std': np.arange(1e-2, 0.5, 1e-3)},
        n_iter=500,
        cv=3,
        scoring=scorer,
    )
    random_search.fit(X_train, y_train)
    report(random_search.cv_results_)
    
    
    # resultados_mlp[0,0] = "Lag"
    # resultados_mlp[0,1] = "R-Pearson treino crossval"
    # resultados_mlp[0,2] = "R-Pearson teste crossval"
    # resultados_mlp[0,2] = "R-Pearson teste"
    # resultados_mlp[0,3] = "MSE treino crossval"
    # resultados_mlp[0,4] = "MSE teste crossval"
    # resultados_mlp[0,5] = "MAE treino crossval"
    # resultados_mlp[0,6] = "MAE teste crossval"
    # resultados_mlp[0,7] = "MSE teste"
    # resultados_mlp[0,8] = "MAE teste"
    print("Criando array de resultados")
    resultados_grnn[lags,0] = lags
    resultados_grnn[lags,1] = grnn_r2_train.mean()
    resultados_grnn[lags,2] = grnn_r2_test.mean()   
    resultados_grnn[lags,3] = grnn_mse_train.mean()
    resultados_grnn[lags,4] = grnn_mse_test.mean()
    resultados_grnn[lags,5] = grnn_mae_train.mean()
    resultados_grnn[lags,6] = grnn_mae_test.mean()
    
    
    
    print(resultados_grnn)

    
    
    
    from sklearn.linear_model import LinearRegression
    rl = LinearRegression().fit(X_train,y_train)
    rl_trainscore =rl.score(X_train, y_train)
    rl_testscore=rl.score(X_test, y_test)
    rl_predicttest =rl.predict(X_test)
    rl_predicttrain =rl.predict(X_train)
    rl_r2=pearsonr(y_test, rl_predicttest)

    
    print("Iniciando Regressão Linear")

    rl_r2_train = cross_val_score(rl, X_train, y_train, cv=splits, scoring=my_scorer)
    rl_r2_test = cross_val_score(rl, X_test, y_test, cv=splits, scoring=my_scorer)
    rl_mse_train = cross_val_score(rl, X_train, y_train, cv=splits, scoring='neg_mean_squared_error')
    rl_mae_train = cross_val_score(rl, X_train, y_train, cv=splits, scoring='neg_mean_absolute_error')
    
    rl_mse_test = cross_val_score(rl, X_test, y_test, cv=splits, scoring='neg_mean_squared_error')
    rl_mae_test = cross_val_score(rl, X_test, y_test, cv=splits, scoring='neg_mean_absolute_error')
    
    rl_mse_predict=mean_squared_error(y_test,rl_predicttest)
    rl_mae_predict=mean_absolute_error(y_test,rl_predicttest)
    
    print("Criando array de resultados da regressão")
    
    resultados_rl[lags,0] = lags
    resultados_rl[lags,1] = rl_r2_train.mean()
    resultados_rl[lags,2] = rl_r2_test.mean()
    resultados_rl[lags,3] = rl_r2[0]
    resultados_rl[lags,4] = rl_mse_train.mean()
    resultados_rl[lags,5] = rl_mse_test.mean()
    resultados_rl[lags,6] = rl_mae_train.mean()
    resultados_rl[lags,7] = rl_mae_test.mean()
    resultados_rl[lags,8] = rl_mse_predict
    resultados_rl[lags,9] = rl_mae_predict
    print(resultados_rl)
    
    
    rl_predicttest = scaler.inverse_transform(rl_predicttest.reshape(-1,1))
    grnn_predicttest = scaler.inverse_transform(prediction.reshape(-1,1))
    y_test = scaler.inverse_transform(y_test.reshape(-1,1))
    plt.title("LAG " + str(lags))
    plt.plot(y_test, label ='Observado', color='orange')
    plt.plot(rl_predicttest, label ='Previsão para dados de teste usando OLS', color='red')
    plt.plot(grnn_predicttest, label='Previsão para dados de teste usando MLP', color='blue')
    plt.legend(loc='best')
    plt.savefig('Grafico Lag' + str(lags) +'.png')

resultados_grnn.tofile("Resultados GRNN.csv", sep=';')
resultados_rl.tofile("Resultados RL.csv", sep=';')
print("Salvando resultados no arquivo")
    


plt.figure()
plt.plot(resultados_grnn[:,0], resultados_grnn[:,1], color='blue', label="R-Pearson para GRNN" )
plt.plot(resultados_rl[:,0], resultados_rl[:,1], color='red', label="R-Pearson para OLS" )
plt.xlabel("LAG")
plt.ylabel("R-Pearson treino")
plt.title("R-Pearson OLS x R-Pearson GRNN")
plt.legend(loc='best')
plt.savefig('Grafico R-PEARSON TREINO.png')
plt.show()
plt.close()

plt.figure()
plt.plot(resultados_grnn[:,0], resultados_grnn[:,2], color='blue', label="R-Pearson para GRNN" )
plt.plot(resultados_rl[:,0], resultados_rl[:,2], color='red', label="R-Pearson para OLS" )
plt.xlabel("LAG")
plt.ylabel("R-Pearson teste")
plt.title("R-Pearson OLS x R-Pearson GRNN")
plt.legend(loc='best')
plt.savefig('Grafico R-PEARSON TESTE.png')
plt.show()
plt.close()


plt.figure()
plt.plot(resultados_grnn[:,0], resultados_grnn[:,3], color='blue', label="MSE para GRNN" )
plt.plot(resultados_rl[:,0], resultados_rl[:,4], color='red', label="MSE para OLS" )
plt.xlabel("LAG")
plt.ylabel("MSE treino")
plt.title("MSE OLS x MSE GRNN")
plt.legend(loc='best')
plt.savefig('Grafico MSE TREINO.png')
plt.show()
plt.close()

plt.figure()
plt.plot(resultados_grnn[:,0], resultados_grnn[:,4], color='blue', label="MSE para GRNN" )
plt.plot(resultados_rl[:,0], resultados_rl[:,8], color='red', label="MSE para OLS" )
plt.xlabel("LAG")
plt.ylabel("MSE teste")
plt.title("MSE OLS x MSE GRNN")
plt.legend(loc='best')
plt.savefig('Grafico MSE TESTE.png')
plt.show()
plt.close()

plt.figure()
plt.plot(resultados_grnn[:,0], resultados_grnn[:,5], color='blue', label="MAE para GRNN" )
plt.plot(resultados_rl[:,0], resultados_rl[:,5], color='red', label="MAE para OLS" )
plt.xlabel("LAG")
plt.ylabel("MAE treino")
plt.title("MAE OLS x MAE GRNN")
plt.legend(loc='best')
plt.savefig('Grafico MAE TREINO.png')
plt.show()
plt.close()

plt.figure()
plt.plot(resultados_grnn[:,0], resultados_grnn
         [:,6], color='blue', label="MAE para GRNN" )
plt.plot(resultados_rl[:,0], resultados_rl[:,9], color='red', label="MAE para OLS" )
plt.xlabel("LAG")
plt.ylabel("MAE teste")
plt.title("MAE OLS x MAE GRNN")
plt.legend(loc='best')
plt.savefig('Grafico MAE TESTE.png')
plt.show()
plt.close()


