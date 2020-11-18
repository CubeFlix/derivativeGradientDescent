# Gradient Descent using Derivatives - Written by Kevin Chen

import numpy as np
import math
import sys
import copy

def cost(func, theta, X, Y):
	# MSE Loss function
	return math.sqrt(sum((func(theta, X) - Y) ** 2) / len(X))

def updateTheta(theta, func, X, Y, cost, learningRate):
	# Updates thetas based on gradient
	# Calculate current loss
	currentLoss = cost(func, theta, X, Y)
	# Calculate gradients/derivitives
	gradients = []
	for i in range(len(theta)):
		# Find partial derivative with respect to theta[i]
		newThetas = copy.deepcopy(theta)
		newThetas[i] += learningRate
		gradients.append((cost(func, newThetas, X, Y) - currentLoss) / learningRate)
	# Return theta moved by gradient value
	return theta - np.array(gradients) * (learningRate * currentLoss) * 2

def updateOptimizeTheta(theta, cost, learningRate):
	# Updates thetas based on gradient
	# Calculate current loss
	currentLoss = cost(theta)
	# Calculate gradients/derivitives
	gradients = []
	for i in range(len(theta)):
		# Find partial derivative with respect to theta[i]
		newThetas = copy.deepcopy(theta)
		newThetas[i] += learningRate
		gradients.append((cost(newThetas) - currentLoss) / learningRate)
	# Return theta moved by gradient value
	return theta - np.array(gradients) * (learningRate * currentLoss) * 2

def train(X, Y, func, numOfThetas, cost=cost, theta=None, learningRate=1e-3, iterations=10000, verbose=False):
	# Trains/fits theta values to X and Y using the given prediction and cost functions
	# Set up thetas
	if theta == None:
		theta = np.random.randn(numOfThetas)

	if verbose:
		print("Training...")
	# Begin training
	for i in range(iterations):
		# Update theta
		theta = updateTheta(theta, func, X, Y, cost, learningRate)

		if verbose:
			# Show debugging data
			outputData = "Theta: " + str(theta) + " Cost: " + str(cost(func, theta, X, Y))
			sys.stdout.write(outputData)
			sys.stdout.flush()
			sys.stdout.write("\b"*len(outputData))

	# Return final thetas
	return theta

def optimize(cost, numOfThetas, theta=None, learningRate=1e-3, iterations=10000, verbose=False):
	# Optimizes/fits theta values to minimize a certain cost function
	# Set up thetas
	if theta == None:
		theta = np.random.randn(numOfThetas)

	if verbose:
		print("Optimizing...")
	# Begin optimizing
	for i in range(iterations):
		# Update theta
		theta = updateOptimizeTheta(theta, cost, learningRate)

		if verbose:
			# Show debugging data
			outputData = "Theta: " + str(theta) + " Cost: " + str(cost(theta))
			sys.stdout.write(outputData)
			sys.stdout.flush()
			sys.stdout.write("\b"*len(outputData))

	# Return final thetas
	return theta

if __name__ == '__main__':
	pass
