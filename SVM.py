# coding: utf-8
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import time

def svm_loss_func(alpha, y, gram):
    alpha_y = alpha * y
    return 0.5 * np.sum(np.outer(alpha_y, alpha_y) * gram) - np.sum(alpha)
    
def svm_loss_gr(alpha, y, gram):
    alpha_y = alpha * y
    return 0.5 * y * np.sum(alpha_y*gram, axis=1) - 1

class SVMPredictor(object):
    def __init__(self, kernel, sp_alpha, sp_X, sp_y, b):
        self._kernel = kernel
        self._sp_alpha = sp_alpha
        self._sp_X = sp_X
        self._sp_y = sp_y
        self._b = b
        
    def predict(self, x):
        K_vec = np.array([self._kernel(x, x_i) for x_i in self._sp_X])
        return np.sign(np.sum(self._sp_alpha * self._sp_y * K_vec) + self._b)
        

class SVM(object):
    def __init__(self, kernel='inner', C=100, alpha_tol=1e-4, g_tol=1e-4):
        self._kernel = lambda x, z: (np.dot(x, z) + 1)**2
        # upper bound for alpha, C is larger if the penelty for misclassified points is bigger
        self._C = C
        # tolerance for alpha and g(x) when checking KKT
        self._alpha_tol = alpha_tol
        self._g_tol = g_tol
    
    def _calc_gram(self, X):
        m, n = X.shape
        gram = np.zeros((m, m))
        for i in range(m):
            for j in range(i, m):
                K = self._kernel(X[i], X[j])
                gram[i, j] = K
                gram[j, i] = K
        return gram
    
    def _calc_eta(self, gram):
        m = gram.shape[0]
        eta = np.zeros((m, m))
        for i in range(m):
            for j in range(i, m):
                eta_ij = gram[i, i]+gram[j, j]-2*gram[i, j]
                eta[i, j] = eta_ij
                eta[j, i] = eta_ij
        return eta    
    
    def _calc_multipliers(self, y, X):
        m, n = X.shape
        gram = self._calc_gram(X)
        
        cons = {'type': 'eq',
                'fun':  lambda alpha, y: np.dot(alpha, y),
                'jac':  lambda alpha, y: y,
                'args': (y, )}
        
        alpha0 = np.zeros(m)
        res = minimize(svm_loss_func, alpha0, args=(y, gram), jac=svm_loss_gr, 
                       constraints=cons, bounds=[(0, self._C) for _ in range(m)],
                       method='SLSQP')
        return res.x
    
    def _calc_multipliers_smo(self, y, X):
        m, n = X.shape
        gram = self._calc_gram(X)
        eta = self._calc_eta(gram)
        
        idx = np.arange(m)
        alpha = np.zeros(m)
        b = 0
        
        for _ in range(5000):   
            alpha_eq0_indic = alpha < self._alpha_tol
            alpha_eqC_indic = alpha > self._C-self._alpha_tol
            alpha_02C_indic = -(alpha_eq0_indic | alpha_eqC_indic)
            # gram matrix selected with alpha>0
            sp_gram = gram[-alpha_eq0_indic]
            # update g, E, etc
            # g is the vector of predicted y, before signed
            g = np.sum(alpha[-alpha_eq0_indic] * y[-alpha_eq0_indic] * sp_gram.T, axis=1) + b
            E = g - y
            yg = y * g
            
            if not _ % 50:
                y_hat = np.sign(g)
                print _, np.sum(y==y_hat)
            
            yg_lw1_indic = yg <= 1-self._g_tol
            yg_up1_indic = yg >= 1+self._g_tol
            
            # samples indicator for these deny KKT
            deny_KKT_indic = (alpha_eq0_indic & yg_lw1_indic) | (alpha_02C_indic & (yg_lw1_indic | yg_up1_indic)) | (alpha_eqC_indic & yg_up1_indic)
            if not deny_KKT_indic.any():
                break
            
            # prepare these rows which deny KKT
            deny_KKT_idx = idx[deny_KKT_indic]
            E_deny_KKT = np.reshape(E, (m, 1))[deny_KKT_indic]
            y_deny_KKT = np.reshape(y, (m, 1))[deny_KKT_indic]
            alpha_deny_KKT = np.reshape(alpha, (m, 1))[deny_KKT_indic]
            
            step_matrix = np.where(eta[deny_KKT_indic]==0, 0, (E_deny_KKT-E) / eta[deny_KKT_indic])
            # potential step length for j, before truncation
            step_matrix4j =  y * step_matrix
            # new_alpha_j before trancation
            new_alpha_matrix4j = alpha + step_matrix4j
            
            # calculate the tracation bound for new_alpha_j
            y_eq_matrix = (y_deny_KKT-y == 0)
            d_alpha_matrix = alpha - alpha_deny_KKT
            L_ineq = np.where(d_alpha_matrix < 0, 0, d_alpha_matrix)
            H_ineq = np.where(d_alpha_matrix < 0, d_alpha_matrix, 0) + self._C
            s_alpha_matrix = alpha + alpha_deny_KKT
            L_eq   = np.where(s_alpha_matrix < self._C, 0, s_alpha_matrix - self._C)
            H_eq   = np.where(s_alpha_matrix < self._C, s_alpha_matrix, self._C)
            L = np.where(y_eq_matrix, L_eq, L_ineq)
            H = np.where(y_eq_matrix, H_eq, H_ineq)
            
            # new_alpha_j after trancation
            new_alpha_matrix4j = np.where(new_alpha_matrix4j < L, L, new_alpha_matrix4j)
            new_alpha_matrix4j = np.where(new_alpha_matrix4j > H, H, new_alpha_matrix4j)
            # potential step length for j, after truncation
            step_matrix4j = new_alpha_matrix4j - alpha
            
            # select the maximun step
            abs_step_matrix = np.abs(step_matrix4j)
            vec_idx = np.argmax(abs_step_matrix)
            i_step, j_step = vec_idx/m, vec_idx%m
            if abs_step_matrix[i_step, j_step] == 0:
                break            
            
            # back to full matrix index, i.e. (m, m)
            i, j = deny_KKT_idx[i_step], j_step
            
            # new alpha
            alpha_j = new_alpha_matrix4j[i_step, j_step]            
            alpha_i = alpha[i] + y[i]*y[j]*(alpha[j] - alpha_j)
            
            # update bias
            b_i = -E[i] - y[i]*gram[i, i]*(alpha_i-alpha[i]) - y[j]*gram[j, i]*(alpha_j-alpha[j]) + b
            b_j = -E[j] - y[i]*gram[i, j]*(alpha_i-alpha[i]) - y[j]*gram[j, j]*(alpha_j-alpha[j]) + b
            if self._alpha_tol < alpha_i < self._C-self._alpha_tol:
                b = b_i
            elif self._alpha_tol < alpha_j < self._C-self._alpha_tol:
                b = b_j
            else:
                b = (b_i + b_j) / 2.0
            
            # update alpha
            alpha[i] = alpha_i
            alpha[j] = alpha_j
        
        return alpha
    
    
    def _construct_predictor(self, y, X):
        alpha = self._calc_multipliers(y, X)
        sp_indic = alpha > self._alpha_tol
        
        print sum(sp_indic)        
        
        sp_alpha = alpha[sp_indic]
        sp_X = X[sp_indic]
        sp_y = y[sp_indic]
        
        idx = np.argmax(sp_alpha)
        K_vec = np.array([self._kernel(sp_X[idx], x_i) for x_i in sp_X])
        b = sp_y[idx] - np.sum(sp_alpha * sp_y * K_vec)
        return SVMPredictor(self._kernel, sp_alpha, sp_X, sp_y, b)
        

if __name__ == '__main__':
    X = np.array([[3, 3],
                  [3, 4],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([1, 1, -1, -1, -1])
    svm = SVM()
    
    print svm._calc_multipliers_smo(y, X)
    
    
    np.random.seed(1226)
    n = 500
    
    df = pd.DataFrame({'x1': np.random.randn(n),
                       'x2': np.random.randn(n),
                       'e' : np.random.randn(n)})
    df['y'] = np.where(df['x1']**2+df['x2']+0.2*df['e'] > 1, 1, -1)
    
    svm = SVM()
#    
#    
#    g = sns.FacetGrid(df, hue="y", size=8)
#    g.map(plt.scatter, "x1", "x2", alpha=.7)
#    g.add_legend()
#    plt.show()
#    
#    
    X = np.array(df[['x1', 'x2']])
    y = np.array(df['y'])
    alpha = svm._calc_multipliers_smo(y, X)
    
    
#    svmp = svm._construct_predictor(y, X)
#    for x_i, y_i in zip(X, y):
#        y_hat = svmp.predict(x_i)
#        if y_i != y_hat:
#            print x_i, np.sum(x_i**2)
    
    
    
    