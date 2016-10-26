=======
 SVM
=======

By syuoni (https://github.com/syuoni)

--------------
 Introduction
--------------

A implementation of Support Vector Machine (SVM) with Python. 

Since a deeply vectorized impelementation of Sequential Minimal Optimization (SMO) algrithm, the solution is quite fast. 

--------------
 Demonstration
--------------
For example,

::

    >>> from SVM import SVM
    >>> svm = SVM()
    >>> predictor = svm._construct_predictor(y, X, method='smo')
    >>> predictor.predict(x_test)
