
# coding: utf-8

# In[1]:


# Import libraries
import warnings
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Defaults
plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.rcParams.update({'font.size': 12})
plt.style.use('ggplot')


# In[2]:


# Load the data
data = pd.read_csv('Time_Hour_air_PM2.5.csv', engine='python', skipfooter=3)
# A bit of pre-processing to make it nicer
data['Time']=pd.to_datetime(data['Time'], format='%Y-%m-%d')
data.set_index(['Time'], inplace=True)

# Plot the data
data.plot()
plt.ylabel('Hour of the day')
plt.xlabel('Hour')
plt.show()


# In[3]:


# Define the d and q parameters to take any value between 0 and 1
q = d = range(0, 2)
# Define the p parameters to take any value between 0 and 3
p = range(0, 4)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 3) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('ARIMA: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('ARIMA: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('ARIMA: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('ARIMA: {} x {}'.format(pdq[2], seasonal_pdq[4]))


# In[13]:


train_data = data['1819-01-01':'1829-12-01']
test_data = data['1830-01-01':'1830-12-01']


# In[5]:


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

            print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic), end='\r')
            AIC.append(results.aic)
            SARIMAX_model.append([param, param_seasonal])
        except:
            continue


# In[6]:


print('The smallest AIC is {} for model ARIMA{}x{}'.format(min(AIC), SARIMAX_model[AIC.index(min(AIC))][0],SARIMAX_model[AIC.index(min(AIC))][1]))


# In[7]:


# Let's fit this model
mod = sm.tsa.statespace.SARIMAX(train_data,
                                order=SARIMAX_model[AIC.index(min(AIC))][0],
                                seasonal_order=SARIMAX_model[AIC.index(min(AIC))][1],
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()


# In[8]:


results.plot_diagnostics(figsize=(20, 14))
plt.show()


# In[9]:


pred0 = results.get_prediction(start='1828-01-01', dynamic=False)
pred0_ci = pred0.conf_int()


# In[10]:


pred1 = results.get_prediction(start='1828-01-01', dynamic=True)
pred1_ci = pred1.conf_int()


# In[16]:


pred2 = results.get_forecast('1832-12-01')
pred2_ci = pred2.conf_int()
print(pred2.predicted_mean['1830-01-01':'1830-12-01'])


# In[12]:


ax = data.plot(figsize=(20, 16))
pred0.predicted_mean.plot(ax=ax, label='1-step-ahead Forecast (get_predictions, dynamic=False)')
pred1.predicted_mean.plot(ax=ax, label='Dynamic Forecast (get_predictions, dynamic=True)')
pred2.predicted_mean.plot(ax=ax, label='Dynamic Forecast (get_forecast)')
ax.fill_between(pred2_ci.index, pred2_ci.iloc[:, 0], pred2_ci.iloc[:, 1], color='k', alpha=.1)
plt.ylabel('Hour of the day')
plt.xlabel('Hour')
plt.legend()
plt.show()


# In[33]:


prediction = pred2.predicted_mean['1830-01-01':'1830-12-01'].values
# flatten nested list
truth = list(itertools.chain.from_iterable(test_data.values))
# Mean Absolute Percentage Error
MAPE = np.mean(np.abs((truth - prediction) / truth)) * 100

print('The Mean Absolute Percentage Error for the forecast of 1830 is {:.2f}%'.format(MAPE))

