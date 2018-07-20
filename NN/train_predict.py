import sys
import json
import build_model
import data_helper
import numpy as np
from numpy import exp
import pandas as pd
import matplotlib.pyplot as plt

def train_predict():
	"""Train and predict time series data"""

	# Load command line arguments 
	train_file = sys.argv[1]
	parameter_file = sys.argv[2]

	# Load training parameters
	params = json.loads(open(parameter_file).read())

	# print(params)
	# print('================')

	for q in range(29):
		params["window_size"] = q+1

		# Load time series dataset, and split it into train and test
		x_train, y_train, x_test, y_test, x_test_raw, y_test_raw,\
			last_window_raw, last_window = data_helper.load_timeseries(train_file, params)

		# Build RNN (FFNN) model
		FFNN_layer = [1, params['window_size'], params['hidden_unit'], 1]
		print(FFNN_layer)
		print('================')
		model = build_model.rnn_FFNN(FFNN_layer, params)

		# Train RNN (FFNN) model with train set
		model.fit(
			x_train,
			y_train,
			batch_size=params['batch_size'],
			epochs=params['epochs'],
			validation_split=params['validation_split'])

		# Check the model against test set
		predicted = build_model.predict_next_timestamp(model, x_test)        
		predicted_raw = []
		RMSE_result_raw = []
		for i in range(len(x_test_raw)):
			predicted_raw.append((predicted[i] + 1) * x_test_raw[i][0])
			RMSE_result_raw.append((y_test_raw[i] + 1)-(predicted_raw[i] + 1))

		#draw_xticks
		#data*0.2
		x = []
		my_xticks = []
		for o in range(10):
			x.append((4000/10)*(o+1))
			my_xticks.append((2008+o))

		# Plot graph: predicted VS actual
		plt.figure(facecolor='white')
		plt.subplot(111)
		plt.plot(predicted_raw, color='red', label='Actual')
		plt.plot(y_test_raw, color='blue', label='Predicted')	
		plt.xticks(x, my_xticks)
		plt.title('predicted VS actual')
		#plt.show()
		plt.savefig("p-predicted/predicted-"+str(q+1)+".png")

		# print('================')
		# print(predicted_raw)
		# print(y_test_raw)
		# print(predicted_raw.shape)
		# print(y_test_raw.shape)

		MAPE = (sum((predicted_raw-y_test_raw)/len(predicted_raw)) * 100) / len(predicted_raw)
		# print(MAPE)
		# np.savetxt('MAPE', MAPE)
		# print('================')

		
		# print(RMSE_result_raw)
		# print(predicted_raw)
		# print(y_test_raw)
		# print(type(RMSE_result_raw))
		# print(type(predicted_raw))
		# print(type(y_test_raw))
		#RMSE
		RMSE_result = np.sqrt(sum((predicted_raw-y_test_raw)**2)/len(predicted_raw))
		plt.figure(facecolor='white')
		plt.plot(RMSE_result_raw, color='blue', label='RMSE_result')
		# plt.plot(predicted_raw, color='red', label='predicted')
		# plt.plot(y_test_raw, color='green', label='Original')
		plt.ylabel('PM2.5-Value', fontsize=12)
		plt.xlabel('Year', fontsize=12)
		plt.xticks(x, my_xticks)
		plt.title('RMSE: %.4f'% RMSE_result)
		#plt.show()
		plt.savefig("p-RMSE/draw_RMSE-"+str(q+1)+".png")


		# Predict next time stamp 
		next_timestamp = build_model.predict_next_timestamp(model, last_window)
		next_timestamp_raw = (next_timestamp[0] + 1) * last_window_raw[0][0]
		print('The next time stamp forecasting is: {}'.format(next_timestamp_raw))

		#print(RMSE_result)
		for i in range(15):
			# print(RMSE_result)
			# print(min_RMSE[i])
			if RMSE_result < min_RMSE_value[i]:
				min_RMSE[i] = q+1
				min_RMSE_value[i] = RMSE_result
				min_next_temp[i] = next_timestamp_raw
				MAPE_value[i] = MAPE
				break;

if __name__ == '__main__':
	# python3 train_predict.py ./data/data.csv ./training_config.json
	min_RMSE_value = [999]*5
	min_RMSE = [0]*5
	min_next_temp = [0]*5
	MAPE_value = [0.0]*5
	ab = []
	ab.append(min_RMSE)
	ab.append(min_RMSE_value)
	ab.append(min_next_temp)
	ab.append(MAPE_value)
	train_predict()
	print(min_RMSE)
	print(min_RMSE_value)
	print(min_next_temp)
	print(MAPE_value)
	print('================')
	np.savetxt('log.txt', ab, fmt="%10.4f")

	# // 比如你有1000个数据，这个数据集可能太大了，全部跑一次再调参很慢，
	# // 于是可以分成100个为一个数据集，这样有10份。
	# // batch_size=100

	# // 这100个数据组成的数据集叫batch每跑完一个batch都要更新参数，
	# // 这个过程叫一个iteration

	# // epoch指的就是跑完这10个batch（10个iteration）的这个过程


