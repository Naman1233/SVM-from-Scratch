from my_svm import svm, print_dict
import numpy as np
clf = svm(kernel='linear')
X = np.array([[1,1], [1, 2], [2, 1], [6, 6], [5, 6], [6, 5]])
y = np.array([0, 0, 0, 1, 1, 1])
clf.fit(X, y)
my_dict = {
    "Support_vectors": clf.support_vectors_,
    "Support_vector indices": clf.support_,
    "N_support": clf.n_support_,
    "Dual_coefs": clf._dual_coef_,
    "B": clf._intercept_,
    "ProbA": clf.probA_,
    "Prob": clf.probB_,
    "degree": clf.degree,
    "coef0": clf.coef0,
    "Gamma LR": clf._gamma,
}
print("\n\nMy SVM:")
print("Sample Predictions: ", clf.predict([[5, 5]]), clf.predict([[0, 0]]))
print_dict(my_dict)
from sklearn.svm import SVC
clf = SVC(kernel='linear')
X = np.array([[1,1], [1, 2], [2, 1], [6, 6], [5, 6], [6, 5]])
y = np.array([0, 0, 0, 1, 1, 1])
clf.fit(X, y)
my_dict = {
    "Support_vectors": clf.support_vectors_,
    "Support_vector indices": clf.support_,
    "N_support": clf.n_support_,
    "Dual_coefs": clf._dual_coef_,
    "B": clf._intercept_,
    "ProbA": clf.probA_,
    "Prob": clf.probB_,
    "degree": clf.degree,
    "coef0": clf.coef0,
    "Gamma LR": clf._gamma,
}
print("\n\nSklearn SVM:")
print("Sample Predictions: ", clf.predict([[5, 5]]), clf.predict([[0, 0]]))
print_dict(my_dict)