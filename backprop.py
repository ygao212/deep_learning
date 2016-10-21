import numpy as np
import matplotlib.pyplot as plt

def forward(X, W1, b1, W2, b2):
	Z = 1 / (1 + np.exp(-X.dot(W1) + b1))
	A = Z.dot(W2) + b2
	expA = np.exp(A)
	Y = expA / expA.sum(axis=1, keepdims=True)
	return Y, Z

def classification_rate(Y, P):
	n_correct = 0
	n_total = 0
	for i in xrange(len(Y)):
		n_total += 1
		if Y[i] == P[i]:
			n_correct += 1
	return float(n_correct) / n_total

def derivative_w2(Z, T, Y):
	N, K = T.shape
	M = Z.shape[1]

	# ret1 = np.zeros((M, K))
	# for n in xrange(N):
	# 	for m in xrange(M):
	# 		for k in xrange(K):
	# 			ret1[m,k] += (T[n,k] - Y[n,k]) * Z[n,m]
	# return ret1

	# ret2 = np.zeros((M, K))
	# for n in xrange(N):
	# 	for k in xrange(K):
	# 		ret2[:, k] += (T[n,k] - Y[n,k]) * Z[n,:]
	# assert(np.abs(ret1-ret2).sum() < 10e-10)

	# ret3 = np.zeros((M, K))
	# for n in xrange(N):
	# 	ret3 += np.outer(Z[n], T[n] - Y[n])
	# assert(np.abs(ret2 - ret3).sum() < 10e-10)

	# ret4 = Z.T.dot(T-Y)
	# assert(np.abs(ret4 - ret3).sum() < 10e-10)
	return Z.T.dot(T - Y)

def derivative_b2(T, Y):
	return (T - Y).sum(axis=0)

def derivative_w1(X, Z, T, Y, W2):
	# N, D = X.shape
	# M, K = W2.shape

	# ret1 = np.zeros((D,M))
	# for n in xrange(N):
	# 	for k in xrange(K):
	# 		for m in xrange(M):
	# 			for d in xrange(D):
	# 				ret1[d,m] += (T[n,k] - Y[n,k])*W2[m,k]*Z[n,m]*(1 - Z[n,m])*X[n,d]
	# return ret1
	dZ = (T-Y).dot(W2.T) * Z * (1-Z)
	return X.T.dot(dZ)

def derivative_b1(T, Y, W2, Z):
	return ((T-Y).dot(W2.T) * Z * (1 -Z)).sum(axis = 0)	


def cost(T, Y):
	tot = T * np.log(Y)
	return tot.sum()

def main():
	Nclass = 500
	D = 2 # dimensionary of input 
	M = 3 # hidden layer size
	K = 3 # number of classes

	X1 = np.random.randn(Nclass, D) + np.array([0, -2])
	X2 = np.random.randn(Nclass, D) + np.array([2, 2])
	X3 = np.random.randn(Nclass, D) + np.array([-2, 2])
	X = np.vstack((X1, X2, X3))

	Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
	N = len(Y)
	T = np.zeros((N, K))
	for i in xrange(N):
		T[i, Y[i]] = 1

	plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
	plt.show()

	W1 = np.random.randn(D, M)
	b1 = np.random.randn(M)
	W2 = np.random.randn(M, K)
	b2 = np.random.randn(K)

	learning_rate = 10e-7
	costs = []
	for epoch in xrange(100000):
		output, hidden = forward(X, W1, b1, W2, b2)
		if epoch % 100 == 0:
			c = cost(T, output)
			P = np.argmax(output, axis=1)
			r = classification_rate(Y, P)
			print "cost:", c, "classification_rate:", r
			costs.append(c)

		W2 += learning_rate * derivative_w2(hidden, T, output)
		b2 += learning_rate * derivative_b2(T, output)
		W1 += learning_rate * derivative_w1(X, hidden, T, output, W2)
		b1 += learning_rate * derivative_b1(T, output, W2, hidden)

	plt.plot(costs)
	plt.show()
	
	plt.scatter(X[:,0], X[:,1], c=P, s=100, alpha=0.5)
	plt.show()

	print(classification_)

if __name__ == '__main__' :
	main()


