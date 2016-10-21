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
	return float(n_correct / n_total)

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

	output, hidden = forward(X, W1, b1, W2, b2)
	P = np.argmax(output, axis=1)
	R = classification_rate(Y, P)
	
	plt.scatter(X[:,0], X[:,1], c=P, s=100, alpha=0.5)
	plt.show()

	print(classification_)

if __name__ == '__main__' :
	main()


