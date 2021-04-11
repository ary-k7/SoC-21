#!/usr/bin/env python
# coding: utf-8

# In[152]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
path = r'C:\Users\Hp\Downloads\Appendix_A (1).xlsx'
X = pd.read_excel(path)
X

D=pd.DataFrame(X['d'])
N=pd.DataFrame(X, columns=('x1','x2','x3'))
N = np.array(N)
D = np.array(D)
arr = np.concatenate((N, D), axis=1)

print(arr)
print(arr.shape)

def predict(row, weights):
	activation = weights[0]
	for i in range(len(row)-1):
		activation += weights[i + 1] * float((row[i]))
	return 1.0 if activation >= 0.0 else -1.0
 
# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train, l_rate, n_epoch):
	weights = [np.random.randn(0,1) for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		sum_error = 0.0
		for row in train:
			prediction = predict(row, weights)
			error = float(row[-1]) - prediction
			sum_error += error**2
			weights[0] = weights[0] + l_rate * error
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
		print('epoch=%d, lrate=%.3f, error=%.0f' % (epoch, l_rate, sum_error))
	return weights


l_rate = 0.1
n_epoch = 100
weights = train_weights(arr, l_rate, n_epoch)
print(weights)


# In[ ]:




