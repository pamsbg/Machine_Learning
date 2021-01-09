# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 18:47:19 2020

@author: pamsb
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 18:30:41 2020

@author: pamsb
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
plt.style.use('fivethirtyeight')
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, mean_squared_log_error



df = pd.read_csv('./Vazao_mensal.csv', sep=';', parse_dates=['Date'], index_col='Date')

df.plot(figsize=(12,5))
plt.title('Vazão Mensal Camargos')
plt.show()



df.head()
df.describe()
# df.shape()

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False, figsize=(12,5))
df.hist(ax=ax1)
df.plot(kind='kde', ax=ax2)
plt.show();

decomposition = seasonal_decompose(df['Vazao'], freq=12, model='additive')
plt.rcParams['figure.figsize'] = 12, 5
decomposition.plot()
plt.show();


df['year'] = df.index.year
df['month'] = df.index.month
df_pivot = pd.pivot_table(df, values='Vazao', index='month', columns='year', aggfunc='mean')
df_pivot.plot(figsize=(12,8))
plt.legend().remove()
plt.xlabel('Mês')
plt.ylabel('Vazão Mensal')
plt.show()




df.drop(['year', 'month'], axis=1, inplace=True)


fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14,6), sharex=False, sharey=False)
ax1 = plot_acf(df, lags=50, ax=ax1)
ax2 = plot_pacf(df, lags=50, ax=ax2)
plt.show()


#Determing rolling statistics
rolmean = pd.Series(df['Vazao']).rolling(window=12).mean()
rolstd = pd.Series(df['Vazao']).rolling(window=12).std()

#Plot rolling statistics:
orig = plt.plot(df['Vazao'], color='blue',label='Original')
mean = plt.plot(rolmean, color='red', label='Rolling Mean')
std = plt.plot(rolstd, color='black', label = 'Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show(block=False)



print ('Results of Dickey-Fuller Test:')
dftest = adfuller(df['Vazao'])

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key, value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)


df_diff = df.diff().diff(12)


df_diff.dropna(inplace=True)

print ('Results of Dickey-Fuller Test:')
dftest = adfuller(df_diff['Vazao'])

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key, value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)
plt.plot(df_diff['Vazao'])
plt.title('Vazão Mensal Camargos')
plt.savefig('diffplot')
plt.show()

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14,6), sharex=False, sharey=False)
ax1 = plot_acf(df_diff, lags=50, ax=ax1)
ax2 = plot_pacf(df_diff, lags=50, ax=ax2)
plt.savefig('acfpacf2')
plt.show()


import pmdarima as pm
model = pm.auto_arima(df['Vazao'], d=1, D=1,
                      seasonal=True, m=12, trend='c', 
                      start_p=0, start_q=0, max_order=6, test='adf', stepwise=True, trace=True)


model.summary()

#divide into train and validation set
train = df[:int(0.85*(len(df)))]
test = df[int(0.85*(len(df))):]

#plotting the data
train['Vazao'].plot()
test['Vazao'].plot()

model = SARIMAX(train['Vazao'],order=(1,1,0),seasonal_order=(0,1,1,12))
results = model.fit()
results.summary()


results.plot_diagnostics(figsize=(16, 8))
plt.savefig('modeldiagnostics')
plt.show()

forecast_object = results.get_forecast(steps=len(test))

mean = forecast_object.predicted_mean

conf_int = forecast_object.conf_int()

dates = mean.index



plt.figure(figsize=(16,8))

# Plot past CO2 levels
plt.plot(df.index, df, label='real')

# Plot the prediction means as line
plt.plot(dates, mean, label='predicted')

# Shade between the confidence intervals
plt.fill_between(dates, conf_int.iloc[:,0], conf_int.iloc[:,1],
alpha=0.2)

# Plot legend and show figure
plt.legend()
plt.savefig('predtest')
plt.show()

start=len(train)
end=len(train)+len(test)-1
predictions = results.predict(start=start, end=end, dynamic=False, typ='levels').rename('SARIMA(1,1,0)(0,1,1,12) Predictions')


# Compare predictions to expected values
for i in range(len(predictions)):
    print(f"predicted={predictions[i]:<11.10}, expected={test['Vazao'][i]}")
    
    # Plot predictions against known values
title = 'Vazão Mensal Camargos'
ax = test['Vazao'].plot(legend=True,figsize=(12,6),title=title)
predictions.plot(legend=True)
ax.autoscale(axis='x',tight=True)
#ax.set(xlabel=xlabel, ylabel=ylabel);

r2_score(test['Vazao'], predictions)


evaluation_results = pd.DataFrame({'r2_score': r2_score(test['Vazao'], predictions)}, index=[0])
evaluation_results['mean_absolute_error'] = mean_absolute_error(test['Vazao'], predictions)
evaluation_results['mean_squared_error'] = mean_squared_error(test['Vazao'], predictions)
evaluation_results['mean_absolute_percentage_error'] = np.mean(np.abs(predictions - test['Vazao'])/np.abs(test['Vazao']))*100 

evaluation_results


pred_f = results.get_forecast(steps=60)
pred_ci = pred_f.conf_int()
ax = df.plot(label='Vazão', figsize=(14, 7))
pred_f.predicted_mean.plot(ax=ax, label='Forecast')
# ax.fill_between(pred_ci.index,
#                 pred_ci.iloc[:, 0],
#                 pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Vazão Mensal Camargos')
plt.legend()
plt.show()