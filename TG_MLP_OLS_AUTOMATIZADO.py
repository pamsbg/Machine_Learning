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
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, cross_val_predict
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


resultados_mlp = np.zeros((13,13))

resultados_rl = np.zeros((13,13))




# resultados_rl_r2 = []
# resultados_rl_rmse = []
# resultados_rl_mae =[]
# resultados_rl_crossval=[]
# resultados_mlp_r2 = []
# resultados_mlp_rmse = []
# resultados_mlp_mae =[]
# resultados_mlp_crossval=[]
# resultados_mlp_r2_predict=[]
# resultados_mlp_mae_predict=[]
# resultados_mlp_mse_predict=[]



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
    
    # X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1]))
    # X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1]))
    
    def mycustomscorer(y_test, prediction):
        mycustomscorer, _ = pearsonr(y_test, prediction)
        return mycustomscorer
    
    print("Criando função especial para cáluclo de pearson")
    my_scorer = make_scorer(mycustomscorer, greater_is_better=True)
    
    def create_model():
        #create model  
        model = Sequential()
        model.add(Dense(1,input_dim=lags, activation='softplus'))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='RMSprop')
        return model
    
    # train_predict = model.predict(X_train)
    # test_predict = model.predict(X_test)
    print("Rodando Modelo")
    model = KerasRegressor(build_fn=create_model, epochs=1000, batch_size=10, verbose=0)
    
       
    print("Criando crossval de resultados")
    mlp_r2_train = cross_val_score(model, X_train, y_train, cv=splits, scoring=my_scorer)
    mlp_r2_test = cross_val_score(model, X_test, y_test, cv=splits, scoring=my_scorer)
    mlp_mse_train = cross_val_score(model, X_train, y_train, cv=splits, scoring='neg_mean_squared_error')
    mlp_mae_train = cross_val_score(model, X_train, y_train, cv=splits, scoring='neg_mean_absolute_error')
    mlp_mse_test = cross_val_score(model, X_test, y_test, cv=splits, scoring='neg_mean_squared_error')
    mlp_mae_test = cross_val_score(model, X_test, y_test, cv=splits, scoring='neg_mean_absolute_error')
    # mlp_r2=r2_score(y_test,test_predict)

    print("Alinhando Modelo")
    model.fit(X_train, y_train)
    
    print("Prevendo para dados de teste")
    prediction = model.predict(X_test)
    # calculate Pearson's correlation
    
    mlp_r2_predict, _ = pearsonr(y_test, prediction)
    
    mlp_mse_predict = mean_squared_error(y_test, prediction)
    mlp_mae_predict = mean_absolute_error(y_test, prediction)
    
    
    
    
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
    resultados_mlp[lags,0] = lags
    resultados_mlp[lags,1] = mlp_r2_train.mean()
    resultados_mlp[lags,2] = mlp_r2_test.mean()
    resultados_mlp[lags,3] = mlp_r2_predict
    resultados_mlp[lags,4] = mlp_mse_train.mean()
    resultados_mlp[lags,5] = mlp_mse_test.mean()
    resultados_mlp[lags,6] = mlp_mae_train.mean()
    resultados_mlp[lags,7] = mlp_mae_test.mean()
    resultados_mlp[lags,8] = mlp_mse_predict
    resultados_mlp[lags,9] = mlp_mae_predict
    
    print(resultados_mlp)
    
    # resultados_mlp_r2_predict.append(mlp_r2_predict)
    # resultados_mlp_r2_predict.append(mlp_r2_train)
    # resultados_mlp_r2_predict.append(mlp_r2_test)
    # resultados_mlp_mae_predict.append(mlp_rmse_predict)
    # resultados_mlp_mse_predict.append(mlp_mae_predict)   
    # resultados_mlp_r2.append(mlp_r2_train)
    # resultados_mlp_r2.append(mlp_r2_test)    
    # resultados_mlp_rmse.append(mlp_mse)   
    # resultados_mlp_mae.append(mlp_mae)
    
    
    
    # train_predict_plot =np.empty_like(data)
    # train_predict_plot[:,:]=np.nan
    # train_predict_plot[lags:len(train_predict)+lags,:]=train_predict
    
    
    # test_predict_plot =np.empty_like(data)
    # test_predict_plot[:,:]=np.nan
    # test_predict_plot[len(train_predict)+(lags*2)+1:len(data)-1,:]=test_predict
    
    # plt.plot(data, label='Observado', color='blue')
    # plt.plot(train_predict_plot, label='Previsão para os dados de treino', color='red', alpha=0.5)
    # plt.plot(test_predict_plot, label='Previsão para os dados de teste', color='yellow')
    # plt.legend(loc='best')
    # plt.show
    
    # plt.savefig('Grafico MLP Lag' + str(lags) +'.png')
    
    from sklearn.linear_model import LinearRegression
    rl = LinearRegression().fit(X_train,y_train)
    rl_trainscore =rl.score(X_train, y_train)
    rl_testscore=rl.score(X_test, y_test)
    rl_predicttest =rl.predict(X_test)
    rl_predicttrain =rl.predict(X_train)
    rl_r2=pearsonr(y_test, rl_predicttest)
    # rl_r2=r2_score(y_test,rl_predicttest)
    # rl_rmse = mean_squared_error(y_test, rl_predicttest)
    # rl_mae = mean_absolute_error(y_test, rl_predicttest)
    
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
    mlp_predicttest = scaler.inverse_transform(prediction.reshape(-1,1))
    y_test = scaler.inverse_transform(y_test.reshape(-1,1))
    plt.title("LAG " + str(lags))
    plt.plot(y_test, label ='Observado', color='orange')
    plt.plot(rl_predicttest, label ='Previsão para dados de teste usando OLS', color='red')
    plt.plot(mlp_predicttest, label='Previsão para dados de teste usando MLP', color='blue')
    plt.legend(loc='best')
    plt.savefig('Grafico Lag' + str(lags) +'.png')

resultados_mlp.tofile("Resultados MLP_20.csv", sep=';')
resultados_rl.tofile("Resultados RL_20.csv", sep=';')
print("Salvando resultados no arquivo")
    
# resultados_mlp_r2_predict = np.array(resultados_mlp_r2_predict)
# resultados_mlp_mae_predict = np.array(resultados_mlp_mae_predict)
# resultados_mlp_mse_predict =np.array(resultados_mlp_mse_predict)
# resultados_rl_crossval = np.array(resultados_rl_crossval)    
# resultados_rl_r2 = np.array(resultados_rl_r2)       
# resultados_rl_rmse = np.array(resultados_rl_rmse)        
# resultados_rl_mae=np.array(resultados_rl_mae)    
# resultados_mlp_rmse = np.array(resultados_mlp_rmse)    
# resultados_mlp_r2 = np.array(resultados_mlp_r2)   
# resultados_mlp_mae=np.array(resultados_mlp_mae)
# resultados_mlp_r2_predict.tofile('Resultados R2 MLP predict lag.csv', sep=';')
# resultados_mlp_mae_predict.tofile('Resultados MAE MLP predict lag.csv', sep=';')
# resultados_mlp_mse_predict.tofile('Resultados MSE MLP predict lag.csv', sep=';')
# resultados_mlp_r2.tofile('Resultados R2 MLP lag.csv', sep=';')
# resultados_mlp_rmse.tofile('Resultados RMSE MLP lag.csv', sep=';')
# resultados_mlp_mae.tofile('Resultados MAE MLP lag.csv', sep=';')
# resultados_rl_crossval.tofile('Resultados CrossValScore OLS lag.csv', sep=';')    
# resultados_rl_r2.tofile('Resultados R2 OLS lag.csv', sep=';')
# resultados_rl_rmse.tofile('Resultados RMSE OLS Lag.csv', sep=';') 
# resultados_rl_mae.tofile('Resultados MAE OLS lag.csv', sep=';') 
    # rl_train_predict_plot =np.empty_like(data)
    # rl_train_predict_plot[:,:]=np.nan
    # rl_train_predict_plot[lags:len(rl_predicttrain)+lags,:]=rl_predicttrain
    
    
    # rl_test_predict_plot =np.empty_like(data)
    # rl_test_predict_plot[:,:]=np.nan
    # rl_test_predict_plot[len(rl_predicttest)+(lags*2)+1:len(data)-1,:]=rl_predicttest
    
    # plt.plot(data, label='Observado', color='blue')
    # plt.plot(rl_train_predict_plot, label='Previsão para os dados de treino', color='red', alpha=0.5)
    # plt.plot(rl_test_predict_plot, label='Previsão para os dados de teste', color='yellow')
    # plt.legend(loc='best')
    # plt.show
    
    # plt.savefig('Grafico OLS Lag' + str(lags) +'.png')
# plt.figure()
# plt.plot(resultados_mlp[:,0], resultados_mlp[:,1], color='blue', label="R-Pearson para MLP" )
# plt.plot(resultados_rl[:,0], resultados_rl[:,1], color='red', label="R-Pearson para OLS" )
# plt.xlabel("LAG")
# plt.ylabel("R-Pearson treino")
# plt.title("R-Pearson OLS x R-Pearson MLP")
# plt.legend(loc='best')
# plt.savefig('Grafico R-PEARSON TREINO.png')
# plt.show()
# plt.close()

# plt.figure()
# plt.plot(resultados_mlp[:,0], resultados_mlp[:,2], color='blue', label="R-Pearson para MLP" )
# plt.plot(resultados_rl[:,0], resultados_rl[:,2], color='red', label="R-Pearson para OLS" )
# plt.xlabel("LAG")
# plt.ylabel("R-Pearson teste")
# plt.title("R-Pearson OLS x R-Pearson MLP")
# plt.legend(loc='best')
# plt.savefig('Grafico R-PEARSON TESTE.png')
# plt.show()
# plt.close()

# plt.figure()
# plt.plot(resultados_mlp[:,0], resultados_mlp[:,3], color='blue', label="R-Pearson para MLP" )
# plt.plot(resultados_rl[:,0], resultados_rl[:,3], color='red', label="R-Pearson para OLS" )
# plt.xlabel("LAG")
# plt.ylabel("R-Pearson teste sem crossval")
# plt.title("R-Pearson OLS x R-Pearson MLP")
# plt.legend(loc='best')
# plt.savefig('Grafico R-PEARSON TESTE SEM CROSSVAL.png')
# plt.show()
# plt.close()

# plt.figure()
# plt.plot(resultados_mlp[:,0], resultados_mlp[:,4], color='blue', label="MSE para MLP" )
# plt.plot(resultados_rl[:,0], resultados_rl[:,4], color='red', label="MSE para OLS" )
# plt.xlabel("LAG")
# plt.ylabel("MSE treino")
# plt.title("MSE OLS x MSE MLP")
# plt.legend(loc='best')
# plt.savefig('Grafico MSE TREINO.png')
# plt.show()
# plt.close()

# plt.figure()
# plt.plot(resultados_mlp[:,0], resultados_mlp[:,5], color='blue', label="MSE para MLP" )
# plt.plot(resultados_rl[:,0], resultados_rl[:,5], color='red', label="MSE para OLS" )
# plt.xlabel("LAG")
# plt.ylabel("MSE teste")
# plt.title("MSE OLS x MSE MLP")
# plt.legend(loc='best')
# plt.savefig('Grafico MSE TESTE.png')
# plt.show()
# plt.close()

# plt.figure()
# plt.plot(resultados_mlp[:,0], resultados_mlp[:,6], color='blue', label="MAE para MLP" )
# plt.plot(resultados_rl[:,0], resultados_rl[:,6], color='red', label="MAE para OLS" )
# plt.xlabel("LAG")
# plt.ylabel("MAE treino")
# plt.title("MAE OLS x MAE MLP")
# plt.legend(loc='best')
# plt.savefig('Grafico MAE TREINO.png')
# plt.show()
# plt.close()

# plt.figure()
# plt.plot(resultados_mlp[:,0], resultados_mlp[:,7], color='blue', label="MAE para MLP" )
# plt.plot(resultados_rl[:,0], resultados_rl[:,7], color='red', label="MAE para OLS" )
# plt.xlabel("LAG")
# plt.ylabel("MAE teste")
# plt.title("MAE OLS x MAE MLP")
# plt.legend(loc='best')
# plt.savefig('Grafico MAE TESTE.png')
# plt.show()
# plt.close()


# plt.figure()
# plt.plot(resultados_mlp[:,0], resultados_mlp[:,8], color='blue', label="MSE para MLP" )
# plt.plot(resultados_rl[:,0], resultados_rl[:,8], color='red', label="MSE para OLS" )
# plt.xlabel("LAG")
# plt.ylabel("MSE teste")
# plt.title("MSE OLS x MAE MLP")
# plt.legend(loc='best')
# plt.savefig('Grafico MSE TESTE sem crossval.png')
# plt.show()
# plt.close()


# plt.figure()
# plt.plot(resultados_mlp[:,0], resultados_mlp[:,9], color='blue', label="MAE para MLP" )
# plt.plot(resultados_rl[:,0], resultados_rl[:,9], color='red', label="MAE para OLS" )
# plt.xlabel("LAG")
# plt.ylabel("MAE teste")
# plt.title("MAE OLS x MAE MLP")
# plt.legend(loc='best')
# plt.savefig('Grafico MAE TESTE sem crossval.png')
# plt.show()
# plt.close()