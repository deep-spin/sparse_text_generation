import numpy as np
import matplotlib.pyplot as plt
import pdb
import pickle

with open('p_target', 'rb') as f:
    p_ = pickle.load(f)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

p=np.array(p_)
V = 50257
n_sents = len(p)

lambdas = np.linspace(0, .7, 10000)
losses = np.zeros_like(lambdas)

for i in range(len(lambdas)):
    lambda_ = lambdas[i]
    # Assume groundtruth is 0.
    loss = -np.sum(np.log((1-lambda_)*p + lambda_/V)) / n_sents
    losses[i] = loss

lambda_ = .5
eta = .00001
c = np.ones(n_sents)/n_sents
r = p / (1./V - p)
for iter in range(100000):
    lambda_ = lambda_ + eta* (c/(lambda_ + r)).sum()
    if lambda_ < 0.:
        lambda_ = 0.
    elif lambda_ > 1.:
        lambda_ = 1.
    print(lambda_)


eps = lambda_/(V *(1-lambda_))

lambda_, idx= find_nearest(lambdas, lambda_)


print('perplexity:', np.exp(losses[idx]))

print('optimal perplexity:', eps)


plt.plot(lambdas, losses)
plt.show()