# coding=utf-8
import sys

#arima时序模型

# Import libraries
import numpy as np
import pandas as pd
import statsmodels
import timeit
import warnings
import itertools
import matplotlib.pyplot as plt
import statsmodels.api as sm

# start = timeit.default_timer()
#Your statements here

#参数初始化
discfile = 'arima_data3.xls'
forecastnum = 5

#读取数据，指定日期列为指标，Pandas自动将“日期”列识别为Datetime格式
data = pd.read_excel(discfile, index_col = u'Time')
#data = pd.read_csv(discfile, index_col = u'日期', engine='python')
# print(len(data))

#时序图
plt.figure(facecolor='white')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
data.plot()
plt.ylabel('PM25-Value', fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.savefig("Original.png")
print('finish-1')
plt.show()

plt.figure(facecolor='white')
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(data) #自相关图
plt.savefig("acf.png")
print('finish-2')
plt.show()

#平稳性检测
from statsmodels.tsa.stattools import adfuller as ADF
print(u'The ADF test results of the original sequence are:', ADF(data[u'PM25']))
print('===============ADF-1===================')
#返回值依次为adf、pvalue、usedlag、nobs、critical values、icbest、regresults、resstore


plt.figure(facecolor='white')
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(data) #偏自相关图
plt.savefig("pacf_Beijing_2008-2017.png")
print('finish-3')
plt.show()



D_data=data

#差分后的结果 p = 1.5291955682384625e-29 < 0.5

#判斷ACF and ADF 是否平衡，沒平衡則繼續D_data.diff()
for i in range(2):
	D_data = D_data.diff().dropna()
	print('diff'+str(i))
	print(D_data)
	#print(len(data))
	print(u'The ADF test results for the differential sequence are:', ADF(D_data[u'PM25'])) #平稳性检测
	#print(u'The ADF test results for the differential sequence are:', ADF(D_data[u'PM25'])) #平稳性检测
	print('===============ADF-2-'+str(i)+'===================')
print(D_data)
D_data.columns = [u'PM25']
D_data.plot() #时序图
plt.savefig("difference.png")
print('finish-4')
print(u'The ADF test results for the differential sequence are:', ADF(D_data[u'PM25'])) #平稳性检测
print('===============ADF-2===================')
plt.show()
print(D_data)
print('===================================================')
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(D_data) #自相关图
plt.savefig("after_acf.png")
print('finish-5')
plt.show()
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(D_data) #偏自相关图
plt.savefig("after_pacf.png")
print('finish-6')
plt.show()
print(u'The ADF test results for the differential sequence are:', ADF(D_data[u'PM25'])) #平稳性检测
print('===============ADF-3===================')
stop = timeit.default_timer()
print (stop - start)


#白噪声检验
from statsmodels.stats.diagnostic import acorr_ljungbox
print(u'The white noise test result of the differential sequence is:', acorr_ljungbox(D_data, lags=1)) #返回统计量和p值
print('===============差分序列的白噪声检验结果为===================')

print('===============run ARIMA===================')
from statsmodels.tsa.arima_model import ARIMA

D_data[u'PM25'] = D_data[u'PM25'].astype(float)


pmax = int(len(D_data)/10) #一般阶数不超过length/10
qmax = int(len(D_data)/10) #一般阶数不超过length/10
bic_matrix = [] #bic矩阵
for p in range(pmax+1):
  tmp = []
  for q in range(qmax+1):
    try: #存在部分报错，所以用try来跳过报错。
      tmp.append(ARIMA(D_data, (p,1,q)).fit().bic)
    except:
      tmp.append(None)
  bic_matrix.append(tmp)

bic_matrix = pd.DataFrame(bic_matrix) #从中可以找出最小值

print(bic_matrix)

p,q = bic_matrix.stack().idxmin() #先用stack展平，然后用idxmin找出最小值位置。
print(u'The minimum p and q values for BIC are: %s、%s' %(p,q)) 
print('===============BIC===================')
model = ARIMA(D_data, (p,1,q)).fit() #建立ARIMA(0, 1, 1)模型
print('===============summary2===================')
reload(sys) 
sys.setdefaultencoding('utf-8') 
print(model.summary2()) #给出一份模型报告
print('===============forecast===================')
print(model.forecast(5)) #作为期5天的预测，返回预测结果、标准误差、置信区间。



# Defaults
plt.figure(facecolor='white')
plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.rcParams.update({'font.size': 12})
plt.style.use('ggplot')

# Load the data
data = pd.read_csv('data.csv', engine='python', skipfooter=3)
# A bit of pre-processing to make it nicer
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
ARIMA_model = []
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(train_data,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()
            print('ARIMA{} - AIC:{}'.format(param, results.aic))
            AIC.append(results.aic)
            ARIMA_model.append([param, param_seasonal])
        except:
            continue

# print('AIC')
# print(AIC)
# print('=================')
# print('ARIMA_model')
# print(ARIMA_model)
# print('=================')
print('The smallest AIC is {} for model ARIMA{}'.format(min(AIC), ARIMA_model[AIC.index(min(AIC))][0]))


# Let's fit this model
mod = sm.tsa.statespace.SARIMAX(train_data,
                                order=ARIMA_model[AIC.index(min(AIC))][0],
                                seasonal_order=ARIMA_model[AIC.index(min(AIC))][1],
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()
#print(results)
print('==========================')
#print(mod)
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
print('RMSE_result', RMSE_result)

