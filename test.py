from main import *
import sys
import numpy as np

def test1():
	# Test 1: Gradient descent fitting test data using function f(x) = 3*x**2 + 2*x + 1
	X = np.array([i for i in range(5)])
	Y = np.array([3*i**2 + 2*i + 1 for i in X])
	
	def func(theta, X):
		return X ** 2 * theta[0] + X * theta[1] + theta[2]
	
	theta = train(X, Y, func, 3, verbose=True)
	return theta

def test2():
	# Test 2: Gradient descent fitting test data using function 3*x + 1
	X = np.array([1, 2, 3])
	Y = np.array([4, 7, 10])
	
	def func(theta, X):
		return X * theta[0] + theta[1]

	theta = train(X, Y, func, 2, verbose=True)
	return theta

def test3():
	# Test 3: Custom optimizations using a^2 + b^2 = 25 and a + b = 7
	def customCost(theta):
		lossA = (theta[0] ** 2 + theta[1] ** 2) - 25
		lossB = (theta[0] + theta[1]) - 7
		return (lossA ** 2 + lossB ** 2) / 10
	theta = optimize(customCost, 2, verbose=False)
	return theta

if __name__ == '__main__':
	userIn = sys.argv[1]
	if int(userIn) == 1:
		print(test1())
	elif int(userIn) == 2:
		print(test2())
	elif int(userIn) == 3:
		print(test3())