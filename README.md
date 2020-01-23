# SVM-from-Scratch
This is the implementation of SVM (from scratch) with the help of libsvm (optimizer) and sklearn docs.

This works well with dense metrics and gives the same result as sklearn gives.

# Further improvements
In the my_svm, svm class an additional parameter can also be added to access different modes of optimizers like c_svc and nu_svc.
By we can make more than 10 different combinations and from parameter tuning we can easily converge the right model very easily.

This code supports dense matrics only. For sparse matrix (DOK, LIL) libsvm_sparse will be added soon. 

Test1: Simple example of SVM and compared all the parameters in both (my_svm and sklearn)
Test2: Comparison of both models for a large number of inputs. 
Test3: Real world problems and model comparison.
