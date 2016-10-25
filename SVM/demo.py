# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from SVM import SVM

def demo():
    svm = SVM()
    
    # test for multiplier result
    X = np.array([[3, 3],
                  [3, 4],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([1, 1, -1, -1, -1])
    
    alpha = svm._calc_multipliers_smo(y, X)
    print alpha
    
    # test for nonliner model
    np.random.seed(1226)
    n = 500
    df = pd.DataFrame({'x1': np.random.randn(n),
                       'x2': np.random.randn(n),
                       'e' : np.random.randn(n)})
    df['y'] = np.where(df['x1']**2+df['x2']+0.2*df['e'] > 1, 1, -1)
    
    # g = sns.FacetGrid(df, hue="y", size=8)
    # g.map(plt.scatter, "x1", "x2", alpha=.7)
    # g.add_legend()
    # plt.show()
    
    
    X = np.array(df[['x1', 'x2']])
    y = np.array(df['y'])
    
    svmp0 = svm._construct_predictor(y, X, method='slsqp')
    print 'Error for slsqp:', svmp0.calc_erorr_rate(y, X)
    svmp1 = svm._construct_predictor(y, X, method='smo')
    print 'Error for smo:', svmp1.calc_erorr_rate(y, X)
    
    print 'Error for reality:', (np.sum((np.where(df['x1']**2+df['x2'] > 1, 1, -1) != y)) + 0.0) / n
    

if __name__ == '__main__':
    demo()
