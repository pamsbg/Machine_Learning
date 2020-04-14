# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 16:35:06 2020

@author: pamsb
"""





#redes neurais artificiais


from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score, KFold
from keras.models import Sequential
from sklearn.metrics import accuracy_score
from keras.layers import Dense
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
import numpy as np
import matplotlib.pylab as plt
#matplotlib.pylab inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=15,6
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import math
dateparse=lambda dates: pd.datetime.strptime(dates, '%Y-%m')
df = pd.read_csv('Matriz_vazao_regress.csv', sep=';', parse_dates=['Month'], index_col='Month', date_parser =dateparse)

dataprep=df.iloc[:,13:14]
data=dataprep.values

np.random.seed(3)
data = data.astype('float32')
scaler = MinMaxScaler(feature_range=(0,1))
data = scaler.fit_transform(data)
train = data[:300,:]
test = data[300:,:]

def prepare_data(data, lags):
    X,y = [],[]
    for row in range(len(data)-lags-1):
        a = data[row:(row+lags),0]
        X.append(a)
        y.append(data[row-lags,0])
    return np.array(X),np.array(y)
                  

lags =1 
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

# Function to create model, required for KerasClassifier
def create_model(neurons=1, optimizer='adam'):
	# create model
	model = Sequential()
	model.add(Dense(neurons, input_dim=1, kernel_initializer='uniform', activation='softplus' ))
	model.add(Dense(1, kernel_initializer='uniform', activation='softplus'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer=optimizer)
	return model
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

model = KerasRegressor(build_fn=create_model, epochs=1000, batch_size=10, verbose=1)



# define the grid search parameters
# activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
# batch_size = [10]
# epochs = [1000]
neurons = [1, 5, 10, 15, 20, 25, 30]
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
param_grid = dict(optimizer=optimizer, neurons=neurons)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
    
# train_score=mdl.evaluate(X_train, y_train, verbose=0)
#print('Pontuação de Treino: ' + train_score + 'MSE' + math.sqrt(train_score)+' RMSE')

# test_score=mdl.evaluate(X_test, y_test, verbose=0)
#print('Pontuação de Treino: {:,2f} MSE ({:,2f} RMSE)'.format(test_score, math.sqrt(test_score)))


# train_predict = mdl.predict(X_train)
# test_predict = mdl.predict(X_test)


# r2=r2_score(y_test,test_predict)
# rmse = mean_squared_error(y_test, test_predict)
# mse = mean_absolute_error(y_test, test_predict)



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


