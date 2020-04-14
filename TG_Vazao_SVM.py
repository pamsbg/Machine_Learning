# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 18:30:09 2020

@author: pamsb
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 12:58:14 2020

@author: pamsb
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 12:45:35 2020

@author: pamsb
"""

import numpy as np
import matplotlib.pylab as plt
#matplotlib.pylab inline

import numpy as np
import matplotlib.pylab as plt
#matplotlib.pylab inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=15,6
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import math
dateparse=lambda dates: pd.datetime.strptime(dates, '%Y-%m')
df = pd.read_csv('Matriz_vazao_regress.csv', sep=';', parse_dates=['Month'], index_col='Month', date_parser =dateparse)

dataprep=df.iloc[:,13:14]
data=dataprep.values


np.random.seed(3)
data = data.astype('float32')
# scaler = MinMaxScaler(feature_range=(0,1))
# data = scaler.fit_transform(data)
train = data[:300,:]
test = data[300:,:]

def prepare_data(data, lags=1):
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
#svm

#kernel linear

from sklearn.svm import SVR
regressor_linear = SVR(kernel = 'linear')
regressor_linear.fit(X_train, y_train)
import matplotlib.pyplot as plt
plt.scatter(X_test, y_test)
plt.plot(X_test, regressor_linear.predict(X_test), color = 'red')
regressor_linear.score(X_test, y_test)
# kernel poly
regressor_poly = SVR(kernel = 'poly', degree = 3, gamma = 'auto')
regressor_poly.fit(X_train, y_train)
plt.scatter(X_test, y_test)
plt.plot(X, regressor_poly.predict(X), color = 'red')
regressor_poly.score(X_test, y_test)
# kernel rbf
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
X = scaler_x.fit_transform(X_train)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y_train)
regressor_rbf = SVR(kernel = 'rbf', gamma = 'auto')
regressor_rbf.fit(X_train, y_train)
plt.scatter(X_train, y_train)
plt.plot(X_test, regressor_rbf.predict(X_test), color = 'red')
regressor_rbf.score(X_test, y_test)




