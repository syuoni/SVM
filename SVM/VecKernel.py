# coding: utf-8
import numpy as np

class VecKernel(object):
    @staticmethod
    def linear():
        def linear_kernel_gram(X, Z):
            if len(X.shape) == 1:
                X = X.reshape((1, X.shape[0]))
            if len(Z.shape) == 1:
                Z = Z.reshape((1, Z.shape[0]))
            return np.dot(X, Z.T)
        return linear_kernel_gram
    
    @staticmethod
    def gaussian(sigma=1.0):
        def gaussian_kernel_gram(X, Z, sigma=sigma):
            if len(X.shape) == 1:
                X = X.reshape((1, X.shape[0]))
            if len(Z.shape) == 1:
                Z = Z.reshape((1, Z.shape[0]))
            X2Z2 = np.sum(X**2, axis=1).reshape((X.shape[0], 1)) + np.sum(Z**2, axis=1).reshape((1, Z.shape[0]))
            d_sqr = X2Z2 - 2 * np.dot(X, Z.T)
            return np.exp(-d_sqr / (2*sigma**2))
        return gaussian_kernel_gram
    
    @staticmethod
    def poly(dim=2.0, offset=1.0):
        def poly_kernel_gram(X, Z, dim=dim, offset=offset):
            if len(X.shape) == 1:
                X = X.reshape((1, X.shape[0]))
            if len(Z.shape) == 1:
                Z = Z.reshape((1, Z.shape[0]))
            return (np.dot(X, Z.T) + offset) ** dim
        return poly_kernel_gram
