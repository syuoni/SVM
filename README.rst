=======
 SVM
=======

By syuoni (https://github.com/syuoni)

--------------
 Introduction
--------------

A implementation of Support Vector Machine (SVM) with Python. 

Since a deeply vectorized impelementation of Sequential Minimal Optimization (SMO) algrithm, the computation is quite fast. An alternative optimization method of SLSQP (from scipy) is provided. 

--------------
 Demonstration
--------------
For example,

::

    >>> from SVM import SVM, VecKernel
    >>> svm = SVM(SVM(kernel_gram=VecKernel.gaussian())
    >>> predictor = svm._construct_predictor(y, X, method='smo')
    >>> predictor.predict(x_test)
