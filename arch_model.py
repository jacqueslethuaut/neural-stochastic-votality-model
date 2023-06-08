
"""
Author: Jacques LE THUAUT
Title: Implementation of a ARCH model based on Engle
File: arch_model.py
"""

from scipy.optimize import minimize
import numpy as np

class ArchModel:

    def __init__(self, returns, p, q):
        self.returns = returns
        self.p = p
        self.q = q
        self.n = len(returns)
        self.params = np.random.normal(size=(p+q+1))

    def likelihood(self, params):
        omega = params[0]
        alphas = params[1:1+self.p]
        betas = params[1+self.p:]

        variances = np.full(self.n, 0.1)
        for t in range(max(self.p, self.q), self.n):
            variances[t] = omega
            for i in range(self.p):
                variances[t] += alphas[i]*self.returns[t-i-1]**2
            for i in range(self.q):
                variances[t] += betas[i]*variances[t-i-1]

        errors = self.returns / np.sqrt(variances)
        logliks = -np.log(np.sqrt(2*np.pi*variances)) - errors**2 / 2
        return -np.sum(logliks[1:])

    def fit(self):
        result = minimize(self.likelihood, self.params, options={'disp': True})
        self.params = result.x

    def predict(self, steps):
        variances = np.full(steps, 0.1)
        returns = self.returns.tolist()
        for t in range(steps):
            variances[t] = self.params[0]
            for i in range(self.p):
                variances[t] += self.params[i+1]*returns[-i-1]**2
            for i in range(self.q):
                variances[t] += self.params[i+1+self.p]*variances[-i-1]
            returns.append(np.sqrt(variances[t])*np.random.normal())
        return returns[-steps:]
