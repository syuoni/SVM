# coding: utf-8
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from SVM import SVM, VecKernel

def demo():
    svm = SVM(kernel_gram=VecKernel.linear())
    # test for multiplier result
    X = np.array([[3, 3],
                  [3, 4],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([1, 1, -1, -1, -1])
    alpha = svm._calc_multipliers_smo(y, X)
    print alpha
    
    svm = SVM(kernel_gram=VecKernel.gaussian())
    # test for nonliner model
    np.random.seed(1226)
    n = 500
    df = pd.DataFrame({'x1': np.random.randn(n),
                       'x2': np.random.randn(n),
                       'e' : np.random.randn(n)})
    
    real_bound = df['x1'] + 0.5 * df['x2']**2 - 1
    df['y'] = np.where(real_bound + 0.2 * df['e'] > 0, 1, -1)
    
    # g = sns.FacetGrid(df, hue="y", size=8)
    # g.map(plt.scatter, "x1", "x2", alpha=.7)
    # g.add_legend()
    # plt.show()
    
    X = np.array(df[['x1', 'x2']])
    y = np.array(df['y'])
    
    print 'Error for reality:', (np.sum((np.where(real_bound > 0, 1, -1) != y)) + 0.0) / n
    
    t0 = time.time()
    predictor0 = svm.construct_predictor(y, X, method='slsqp')
    t1 = time.time()
    print 'Time for slsqp:', t1 - t0
    print 'Error for slsqp:', predictor0.calc_erorr_rate(y, X)
    
    t0 = time.time()
    predictor1 = svm.construct_predictor(y, X, method='smo')
    t1 = time.time()
    print 'Time for smo:', t1 - t0
    print 'Error for smo:', predictor1.calc_erorr_rate(y, X)
    
if __name__ == '__main__':
    demo()
