# Real world problems with svm: comparision between my svm model vs skleran's model
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Just to avoid using sklearn 
'''
# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
np.save('Social_Network_Ads_X_train.npy', X_train)
np.save('Social_Network_Ads_X_test.npy', X_test)
np.save('Social_Network_Ads_y_train.npy', y_train)
np.save('Social_Network_Ads_y_test.npy', y_test)
'''

X_train = np.load('Social_Network_Ads_X_train.npy')
X_test = np.load('Social_Network_Ads_X_test.npy')
y_train = np.load('Social_Network_Ads_y_train.npy')
y_test = np.load('Social_Network_Ads_y_test.npy')
#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# Fitting SVM to the Training set
from my_svm import svm
classifier = svm(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

f, ax = plt.subplots(2, 2, figsize=(12, 12))


# Visualising the Training set results

from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
ax[0, 0].contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
ax[0, 0].set_xlim(X1.min(), X1.max())
ax[0, 0].set_ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    ax[0, 0].scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
ax[0, 0].set_title('My SVM (Training set)')
ax[0, 0].set_ylabel('Estimated Salary')
ax[0, 0].set_xlabel('Age')
ax[0, 0].xaxis.set_label_coords(1.05, -0.025)
ax[0, 0].legend()



# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
ax[0, 1].contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
ax[0, 1].set_xlim(X1.min(), X1.max())
ax[0, 1].set_ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    ax[0, 1].scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
ax[0, 1].set_title('My SVM (Test set)')
ax[0, 1].set_xlabel('Age')
ax[0, 1].xaxis.set_label_coords(1.05, -0.025)
ax[0, 1].legend()




from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



# Visualising the Training set results

from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
ax[1, 0].contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
ax[1, 0].set_xlim(X1.min(), X1.max())
ax[1, 0].set_ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    ax[1, 0].scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
ax[1, 0].set_title('SKlearn SVM (Training set)')
ax[0, 0].set_ylabel('Estimated Salary')
ax[0, 1].set_xlabel('Age')
ax[0, 1].xaxis.set_label_coords(1.05, -0.025)
ax[1, 0].legend()



# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
ax[1, 1].contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
ax[1, 1].set_xlim(X1.min(), X1.max())
ax[1, 1].set_ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    ax[1, 1].scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
ax[1, 1].set_title('Sklearn SVM (Test set)')
ax[1, 1].set_xlabel('Age')
ax[1, 1].xaxis.set_label_coords(1.05, -0.025)

ax[1, 1].legend()
f.suptitle("Comparison between svm-scratch from svm-sklearn")
plt.show()