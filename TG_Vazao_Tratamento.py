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
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=15,6
import pandas as pd
data = pd.read_csv("Matriz_vazao_regress.csv", sep=';')

data.columns
data.head()
dateparse=lambda dates: pd.datetime.strptime(dates, '%Y-%m')
date = pd.read_csv('Matriz_vazao_regress.csv', sep=';', parse_dates=['Month'], index_col='Month', date_parser =dateparse)

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
plt.plot(date.index,date.iloc[:,13])

date.describe()
#média da vazão a cada 6 meses em um gráfico
date['vazao'].resample('6M').apply([np.mean]).plot()
sumario = date.groupby('Month')['vazao'].mean().reset_index()
vazao = date['vazao']
pd.Series.rolling(vazao, window=25).mean().plot(style='-g')

optimal_n = None

best_mse = None

db = data[['vazao']].values.astype('float32')

mean_results_for_all_possible_n_values = np.zeros(int(len(db) / 2 - 2))


for n in range(3, int(len(db) / 2 + 1)):
    mean_for_n = np.zeros(len(db) - n)
    for i in range(0, len(db) - n):
        mean_for_n[i] = np.power(np.mean(db[:, 0][i:i+n]) - db[i + n][0], 2)
        
    mean_results_for_all_possible_n_values[n - 3] = np.mean(mean_for_n)
optimal_n = np.argmin(mean_results_for_all_possible_n_values) + 3
best_mse = np.min(mean_results_for_all_possible_n_values)


print("MSE = %s" % mean_results_for_all_possible_n_values)
print("Melhor MSE = %s" % best_mse)
print("Otimo n = %s" % optimal_n)

forecast = np.zeros(len(db) + 1)
for i in range(0, optimal_n):
    forecast[i] = db[i][0]
for i in range(0, len(db) - optimal_n + 1):
        forecast[i+optimal_n] = np.mean(db[:, 0][i:i+optimal_n])
plt.plot(db[:, 0],label = 'Dados Originais')
plt.plot(forecast, label = 'Previsão')
plt.legend()
plt.show()

date['vazao'].head()

from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(date['vazao'], model='multiplicative')
result.plot()

plt.show()


result2 = seasonal_decompose(date['vazao'], model='aditive')
result2.plot()
plt.show()




from statsmodels.tsa.stattools import adfuller
X = date['vazao']
result = adfuller(X)
result
print('ADF Estatíticas: %f' % result[0])
print('Valor de P: %f' % result[1])
print('Valores Críticos:')
for key, value in result[4].items():
   print('\t%s: %.3f' % (key, value))
   
y = date['vazao']
   
   
def adf_test(y):
    # perform Augmented Dickey Fuller test
    print('Resultado do Teste Dickey-Fuller:')
    dftest = adfuller(y, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Teste', 'Valor p', '# de lags', '# de observações'])
    for key, value in dftest[4].items():
        dfoutput['Valores Críticos ({})'.format(key)] = value
    print(dfoutput)
    
adf_test(y)
y_diff = np.diff(y)
plt.plot(y_diff)
adf_test(y_diff)
y_diff2 = np.diff(y_diff)
plt.plot(y_diff2)
adf_test(y_diff2)

# remoção de tendência usando regressão

X = [i for i in range(0,len(date['vazao']))]
X = np.reshape( X, (len(X),1))
y = date['vazao'].values

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)

trend = model.predict(X)
plt.plot(y)
plt.plot(trend)
plt.show()   


detrended = [y[i]-trend[i] for i in range(0, len(date['vazao']))]

# plot detrended
plt.plot(detrended)
plt.plot(y)
plt.plot(trend)
plt.show()


#redes neurais artificiais

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
                  

lags =4
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


mdl = Sequential()
mdl.add(Dense(3,input_dim=lags, activation='relu'))
mdl.add(Dense(1))
mdl.compile(loss='mean_squared_error', optimizer='adam')
mdl.fit(X_train, y_train, epochs=2000, batch_size=2,verbose=2)

# train_score=mdl.evaluate(X_train, y_train, verbose=0)
#print('Pontuação de Treino: ' + train_score + 'MSE' + math.sqrt(train_score)+' RMSE')

# test_score=mdl.evaluate(X_test, y_test, verbose=0)
#print('Pontuação de Treino: {:,2f} MSE ({:,2f} RMSE)'.format(test_score, math.sqrt(test_score)))


train_predict = mdl.predict(X_train)
test_predict = mdl.predict(X_test)


r2=r2_score(y_test,test_predict)
rmse = mean_squared_error(y_test, test_predict)
mse = mean_absolute_error(y_test, test_predict)



train_predict_plot =np.empty_like(data)
train_predict_plot[:,:]=np.nan
train_predict_plot[lags:len(train_predict)+lags,:]=train_predict


test_predict_plot =np.empty_like(data)
test_predict_plot[:,:]=np.nan
test_predict_plot[len(train_predict)+(lags*2)+1:len(data)-1,:]=test_predict

plt.plot(data, label='Observado', color='blue')
plt.plot(train_predict_plot, label='Previsão para os dados de treino', color='red', alpha=0.5)
plt.plot(test_predict_plot, label='Previsão para os dados de teste', color='yellow')
plt.legend(loc='best')
plt.show


