from my_svm import svm, print_dict
import numpy as np
import random
import time

def generate_data(size, c1, c2, r=100):

    r1 = np.random.randint(low=c1[0]-(r//2), high=c1[0] + (r//2), size=(size, 2))
    r2 = np.random.randint(low=c2[0]-(r//2), high=c2[0] + (r//2), size=(size, 2))
    X = np.concatenate([r1, r2], axis=0)
    y = np.concatenate([np.zeros((size,)), np.ones(size,)])
    return X, y

size = 500000
r = 10000000
c1 = [0, 0]
c2 = [200000000, 20]
X, y = generate_data(size, c1, c2, r)
color = {
    0:'b',
    1:'r',
}

clf = svm(kernel='linear')
tic = time.time()
clf.fit(X, y)
tac = time.time()
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
print("Time to fit: ", tac-tic)
print("Sample Predictions: ", clf.predict([[5, 5]]), clf.predict([[0, 0]]))
print_dict(my_dict)



# Sklearn part
from sklearn.svm import SVC
clf = SVC(kernel='linear')
tic = time.time()
clf.fit(X, y)
tac = time.time()
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
print("Time to fit: ", tac-tic)
print("Sample Predictions: ", clf.predict([[5, 5]]), clf.predict([[0, 0]]))
print_dict(my_dict)
# Try only if size is too low to visualize it
'''
import matplotlib.pyplot as plt
from matplotlib.style import use
plt.scatter(X[:size, 0], X[:size, 1], c=color[0])
plt.scatter(X[size:, 0], X[size:, 1], c=color[1])
plt.show()'''