# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pandas import Series
from matplotlib import pyplot
# series = Series.from_csv('Beijing_2008.csv', header=0)
data2 = []
for j in range(10):
	if j >= 2:
		b = 'Beijing_20'+str(j+8)
	else:
		b = 'Beijing_200'+str(j+8)
	with open(b+'.csv','r', encoding='utf-8') as f:
		f.readline()
		f.readline()
		f.readline()
		f.readline()
		data = f.read().splitlines()
	#print(data)
	a = list(map(lambda x: (x.split(',')[7]), data))
	data1 = []
	# print(a)
# original
	# data2.append(str(i[0:10]))
	for i in a:
		if i != '':
			data2.append(int(i))

# data first-order-linear-interpolation-method	
	# temp_1=0 
	# temp_2=1
	# for i in a:
		# if i != '':
		# 	if i != '-999':
		# 		temp_1 += int(i)
		# 		temp_2 += 1
		# 		data2.append(int(i))
		# 		# print(temp_1)
		# 		# print(temp_2)
		# 		# print(temp_1/temp_2)
		# 		# if int(i) > 500:
		# 		# 	print(i)
		# 	else:
		# 		i = temp_1/temp_2
		# 		data2.append(int(i))


x ='Original_'

	print each year data
		print(data1)
		plt.plot(data1)
		if j >= 2:
			plt.title(r'Beijing_20'+str(j+8)+'_PM2.5-Value', fontsize=20)
		else:
			plt.title(r'Beijing_200'+str(j+8)+'_PM2.5-Value', fontsize=20)
		plt.ylabel('PM2.5-Value', fontsize=12)
		plt.xlabel('Month', fontsize=12)
		x = []
		for o in range(12):
			x.append((len(data1)/12)*(o+1))
		my_xticks = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
		plt.xticks(x, my_xticks)
		plt.savefig(b+".png")
		plt.show()

#print 10 years data
print(data2)
np.savetxt('new-temp.csv', data2,fmt='%s',newline='\n')
plt.plot(data2)
plt.title(r'Original_Beijing_2008-2017_PM2.5-Value', fontsize=20)
plt.ylabel('PM2.5-Value', fontsize=12)
plt.xlabel('Year', fontsize=12)
x = []
my_xticks = []
for o in range(10):
	x.append((len(data2)/10)*(o+1))
	my_xticks.append((2008+o))
plt.xticks(x, my_xticks)
plt.savefig("Original_Beijing_2008-2017.png")
plt.show()


