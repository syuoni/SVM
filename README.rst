# SVM

## A implementation of Support Vector Machine(SVM) with Python. 

## Since a deeply vectorized impelementation of Sequential Minimal Optimization(SMO) algrithm, the solution is quite fast. 
--------

To use this package, simply do::

    >>> from SVM import SVM
    >>> svm = SVM()
    >>> predictor = svm._construct_predictor(y, X, method='smo')
    >>> predictor.predict(x)
