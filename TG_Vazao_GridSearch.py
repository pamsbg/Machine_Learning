# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 10:21:23 2020

@author: pamsb
"""

import pandas as pd
import numpy as np
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
# Function to create model, required for KerasClassifier
def create_model(neurons=1, optimizer='Adam'):
	# create model
	model = Sequential()
	model.add(Dense(neurons, input_dim=1, activation='softplus'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer=optimizer)
	return model
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# load dataset
dateparse=lambda dates: pd.datetime.strptime(dates, '%Y-%m')
df = pd.read_csv('Matriz_vazao_regress.csv', sep=';', parse_dates=['Month'], index_col='Month', date_parser =dateparse)
# split into input (X) and output (Y) variables
dataprep=df.iloc[:,13:14]
data=dataprep.values
data = data.astype('float32')
scaler = MinMaxScaler(feature_range=(0,1))
data = scaler.fit_transform(data)
train = data[:300,:]
test = data[300:,:]
# create model
model = KerasRegressor(build_fn=create_model, epochs=1000, batch_size=10, verbose=1)
# define the grid search parameters
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
# optimizer = ['Adadelta']
neurons = [1, 5, 10, 15, 20, 25, 30]
param_grid = dict(neurons=neurons, optimizer=optimizer)
#grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
# param_grid = dict(optimizer=optimizer, )
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(train, train)
prediction = grid.predict(test)


# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))