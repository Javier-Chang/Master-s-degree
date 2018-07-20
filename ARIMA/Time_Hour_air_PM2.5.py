
# coding: utf-8

# Import libraries
import warnings
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Defaults
plt.figure(facecolor='white')
plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.rcParams.update({'font.size': 12})
plt.style.use('ggplot')

# Load the data
data = pd.read_csv('data.csv', engine='python', skipfooter=3)
# A bit of pre-processing to make it nicer
#data = data[:2000]
print(len(data))
data['Time']=pd.to_datetime(data['Time'], format='%Y-%m-%d')
data.set_index(['Time'], inplace=True)

# Plot the data
plt.figure(facecolor='white')
data.plot()
plt.ylabel('Hour of PM2.5')
plt.xlabel('Date')
plt.savefig("Plot_the_data.png")
#plt.show()

# Define the d and q parameters to take any value between 0 and 1
q = d = range(0, 2)
# Define the p parameters to take any value between 0 and 3
p = range(0, 4)


# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Parameter combinations for Seasonal ARIMA...')
print('ARIMA: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('ARIMA: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('ARIMA: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('ARIMA: {} x {}'.format(pdq[2], seasonal_pdq[4]))

train_data = data['2008-01-01':'2016-12-01']
test_data = data['2017-01-01':'2017-12-01']

warnings.filterwarnings("ignore") # specify to ignore warning messages

AIC = []
SARIMAX_model = []
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(train_data,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{} - AIC:-{}'.format(param, results.aic), end='\r')
            AIC.append(results.aic)
            SARIMAX_model.append([param, param_seasonal])
        except:
            continue

# print('AIC')
# print(AIC)
# print('SARIMAX_model')
# print(SARIMAX_model)
print('=================')
print('The smallest AIC is -{} for model ARIMA{}'.format(min(AIC), SARIMAX_model[AIC.index(min(AIC))][0]))


# Let's fit this model
mod = sm.tsa.statespace.SARIMAX(train_data,
                                order=SARIMAX_model[AIC.index(min(AIC))][0],
                                seasonal_order=SARIMAX_model[AIC.index(min(AIC))][1],
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()
print(results)
print('==========================')
# print(mod)
print('==========================')
plt.figure(facecolor='white')
results.plot_diagnostics(figsize=(20, 14))
plt.savefig("plot_diagnostics.png")
#plt.show()

plt.figure(facecolor='white')
pred0 = results.get_prediction(start='2012-01-01', dynamic=False)
pred0_ci = pred0.conf_int()

pred7 = results.get_prediction(start='2012-01-01', dynamic=False)
pred0_ci = pred0.conf_int()
print(pred7.predicted_mean)
np.savetxt('pred7.predicted.csv', pred7.predicted_mean)

pred1 = results.get_prediction(start='2012-01-01', dynamic=True)
pred1_ci = pred1.conf_int()

pred2 = results.get_forecast('2019-12-01')
pred2_ci = pred2.conf_int()
print(pred2.predicted_mean['2017-01-01':'2017-12-01'])

ax = data.plot(figsize=(20, 16))
pred0.predicted_mean.plot(ax=ax, label='1-step-ahead Forecast (get_predictions, dynamic=False)' )
pred1.predicted_mean.plot(ax=ax, label='Dynamic Forecast (get_predictions, dynamic=True)')
pred2.predicted_mean.plot(ax=ax, label='Dynamic Forecast (get_forecast)')
ax.fill_between(pred2_ci.index, pred2_ci.iloc[:, 0], pred2_ci.iloc[:, 1], color='k', alpha=.1)
plt.ylabel('Hour of PM2.5')
plt.xlabel('Date')
plt.legend()
plt.savefig("predicted.png")
#plt.show()
#print('The Mean Absolute Percentage Error for the forecast of year 2017 is 7.32%')
print('==========================')
prediction = pred2.predicted_mean['2017-01-01':'2017-09-01'].values
# flatten nested list
truth = list(itertools.chain.from_iterable(test_data.values))
# print(test_data)
# print(truth)
# print(prediction)
# Mean Absolute Percentage Error
#MAPE = list(np.mean((np.abs((truth - prediction) / truth)) * 100))
MAPE = ((sum((truth - prediction) / truth)) * 100) / 9

print('The Mean Absolute Percentage Error for the forecast of year 2017 is {:.2f}%'.format(MAPE))


#RMSE
np.savetxt('.csv', pred7.predicted_mean)
np.savetxt('pred7.predicted.csv', pred7.predicted_mean)
prediction0 = pred0.predicted_mean['2012-01-01':'2016-12-01'].values
test_data_raw = data['2012-01-01':'2016-12-01'].as_matrix().reshape((60,))
# truth0 = list(itertools.chain.from_iterable(test_data_raw.values))
print(prediction0)
print(test_data_raw)
print(prediction0.shape)
print(test_data_raw.shape)
# print(truth0.shape)
RMSE_result = np.sqrt(sum((test_data_raw-prediction0)**2)/len(test_data_raw))
print(RMSE_result)
        




