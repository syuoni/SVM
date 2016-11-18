# coding: utf-8
from __future__ import division
import numpy as np
from scipy.optimize import minimize

def svm_loss_func(alpha, y, gram):
    alpha_y = alpha * y
    return 0.5 * np.sum(np.outer(alpha_y, alpha_y) * gram) - np.sum(alpha)
    
def svm_loss_gr(alpha, y, gram):
    alpha_y = alpha * y
    return 0.5 * y * np.sum(alpha_y*gram, axis=1) - 1

class SVMPredictor(object):
    def __init__(self, kernel_gram, sp_alpha, sp_X, sp_y, b):
        self._kernel_gram = kernel_gram
        self._sp_alpha = sp_alpha
        self._sp_X = sp_X
        self._sp_y = sp_y
        self._b = b
        
    def predict(self, X):
        sp_gram = self._kernel_gram(X, self._sp_X)
        g = np.sum(self._sp_alpha * self._sp_y * sp_gram, axis=1) + self._b
        return np.sign(g)
    
    def calc_erorr_rate(self, y, X):
        y_hat = self.predict(X)
        return np.sum(y_hat != y) / len(y) 

class SVM(object):
    def __init__(self, kernel_gram, C=100, alpha_tol=1e-4, g_tol=1e-4):
        # vectorized kernel function, which could directly calculate the gram matrix
        self._kernel_gram = kernel_gram
        # upper bound for alpha, C is larger if the penelty for misclassified points is bigger
        self._C = C
        # tolerance for alpha and g(x) when checking KKT
        self._alpha_tol = alpha_tol
        self._g_tol = g_tol
    
    def _calc_eta(self, gram):
        m = gram.shape[0]
        gram_diag = np.diag(gram)
        return gram_diag.reshape((1, m)) + gram_diag.reshape((m, 1)) - 2*gram
    
    def _calc_multipliers_slsqp(self, y, X):
        m, n = X.shape
        gram = self._kernel_gram(X, X)
        
        cons = {'type': 'eq',
                'fun':  lambda alpha, y: np.dot(alpha, y),
                'jac':  lambda alpha, y: y,
                'args': (y, )}
        
        alpha0 = np.zeros(m)
        res = minimize(svm_loss_func, alpha0, args=(y, gram), jac=svm_loss_gr, 
                       constraints=cons, bounds=[(0, self._C) for _ in range(m)],
                       method='SLSQP')
        return res.x
    
    def _calc_multipliers_smo(self, y, X, max_iter=1000, error_tol=5e-4, max_d_error=5, valid_ratio=0.3):       
        if 0 < valid_ratio < 1:
            m, n = X.shape
            sep = round(m * (1 - valid_ratio))
            X_valid = X[sep:]
            y_valid = y[sep:]
            X = X[:sep]
            y = y[:sep]
        else:
            X_valid = X
            y_valid = y
            
        m, n = X.shape        
        gram = self._kernel_gram(X, X)
        eta = self._calc_eta(gram)
        
        idx = np.arange(m)
        alpha = np.zeros(m)
        b = 0
        
        n_iter = 0
        best_alpha = alpha
        min_error_rate = 1.0
        n_d_error = 0
        stop_flag = False
        
        while True:   
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
            
            # Validation
            sp_indic = alpha > self._alpha_tol
            sp_alpha = alpha[sp_indic]
            sp_X = X[sp_indic]
            sp_y = y[sp_indic]
            
            pred = SVMPredictor(self._kernel_gram, sp_alpha, sp_X, sp_y, b)
            error_rate = pred.calc_erorr_rate(y_valid, X_valid)
            if error_rate < min_error_rate:
                if min_error_rate - error_rate < error_tol:
                    n_d_error += 1
                    if n_d_error >= max_d_error:
                        stop_flag = True
                else:
                    n_d_error = 0
                    
                best_alpha = alpha
                min_error_rate = error_rate
                
            n_iter += 1
            if n_iter >= max_iter:
                stop_flag = True
            
            if (not n_iter % (max_iter / 10)) or n_iter == max_iter:
                print n_iter, min_error_rate
            
            if stop_flag:
                break

        return best_alpha
    
    
    def construct_predictor(self, y, X, method='smo', smo_option=None):
        method = method.lower()
        if method == 'smo':
            if smo_option is None:
                alpha = self._calc_multipliers_smo(y, X)
            else:
                alpha = self._calc_multipliers_smo(y, X, **smo_option)
            
        elif method == 'slsqp':
            alpha = self._calc_multipliers_slsqp(y, X)
        else:
            return None
        sp_indic = alpha > self._alpha_tol
        sp_alpha = alpha[sp_indic]
        sp_X = X[sp_indic]
        sp_y = y[sp_indic]
                
        sp_gram = self._kernel_gram(sp_X, sp_X)
        b = np.mean(sp_y - np.sum(sp_alpha * sp_y * sp_gram, axis=1))
        return SVMPredictor(self._kernel_gram, sp_alpha, sp_X, sp_y, b)
