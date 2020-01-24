#Only dense matrix support is available.
'''
Reference: libsvm & sklern official documentation
'''
import numpy as np
import scipy.sparse as sp
import warnings
#just a check
try:
    import libsvm
    f1 = libsvm.set_verbosity_wrap
    del f1
except:
    print("Happened")
    import _libsvm as libsvm

from support import check_X_y, check_random_state, check_array, safe_sparse_dot, check_is_fitted, column_or_1d
from support import check_classification_targets, compute_class_weight

class svm:

    def __init__(self, kernel='rbf', degree=3, gamma='auto_deprecated', coef0=0.0,
                 tol=1e-3, C=1.0, nu=0.0, epsilon=0.0, shrinking=True, probability=False, cache_size=200,
                 class_weight=None, verbose=False, max_iter=-1, random_state=None):
        
        #Learning Rate Can not be zero
        if gamma == 0:
            msg = ("The gamma value of 0.0 is invalid. Use 'auto' to set"
                   " gamma to a value of 1 / n_features.")
            raise ValueError(msg)
        _sparse_kernels = ["linear", "poly", "rbf", "sigmoid", "precomputed"]
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.C = C
        self.nu = nu
        self.epsilon = epsilon
        self.shrinking = shrinking
        self.probability = probability
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = max_iter
        self.random_state = random_state
    
    def fit(self, X, y, sample_weight=None):
        solver_type = 0 # For c_svc
        X, y = self._validate_params(X, y, solver_type, sample_weight=None)
        rnd = check_random_state(self.random_state)
        sample_weight = np.asarray([]
                                   if sample_weight is None
                                   else sample_weight, dtype=np.float64)       
        kernel = self.kernel
        if callable(kernel):
            kernel = 'precomputed'
        fit = self._sparse_fit if self._sparse else self._dense_fit
        if self.verbose:  # pragma: no cover
            print('[LibSVM]', end='')
        seed = rnd.randint(np.iinfo('i').max)
        fit(X, y, sample_weight, solver_type, kernel, random_seed=seed)
        
        self.shape_fit_ = X.shape

        # In binary case, we need to flip the sign of coef, intercept and
        # decision function. Use self._intercept_ and self._dual_coef_
        # internally.
        self._intercept_ = self.intercept_.copy()
        self._dual_coef_ = self.dual_coef_
        if len(self.classes_) == 2:
            self.intercept_ *= -1
            self.dual_coef_ = -self.dual_coef_

        return self


        
    
    def _validate_params(self, X, y, solver_type, sample_weight=None):
        sparse = sp.isspmatrix(X)
        if sparse and self.kernel == "precomputed":
            raise TypeError("Sparse precomputed kernels are not supported.")
        self._sparse = sparse and not callable(self.kernel)
        sample_weight = np.asarray([]
                                   if sample_weight is None
                                   else sample_weight, dtype=np.float64)  
        X, y = check_X_y(X, y, dtype=np.float64,
                         order='C', accept_sparse='csr',
                         accept_large_sparse=False)
        y = self._validate_targets(y)

        # input validation
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have incompatible shapes.\n" +
                             "X has %s samples, but y has %s." %
                             (X.shape[0], y.shape[0]))

        #For a precomputed kernel we require a NxN matrix as X
        if self.kernel == "precomputed" and X.shape[0] != X.shape[1]:
            raise ValueError("X.shape[0] should be equal to X.shape[1]")

        if sample_weight.shape[0] > 0 and sample_weight.shape[0] != X.shape[0]:
            raise ValueError("sample_weight and X have incompatible shapes: "
                             "%r vs %r\n"
                             "Note: Sparse matrices cannot be indexed w/"
                             "boolean masks (use `indices=True` in CV)."
                             % (sample_weight.shape, X.shape))

        if self.gamma in ('scale', 'auto_deprecated'):
            if sparse:
                # var = E[X^2] - E[X]^2
                X_var = (X.multiply(X)).mean() - (X.mean()) ** 2
            else:
                X_var = X.var()
            if self.gamma == 'scale':
                if X_var != 0:
                    self._gamma = 1.0 / (X.shape[1] * X_var)
                else:
                    self._gamma = 1.0
            else:
                self._gamma = 1.0 / X.shape[1]
        elif self.gamma == 'auto':
            self._gamma = 1.0 / X.shape[1]
        else:
            self._gamma = self.gamma
        
        return X, y

    def _validate_targets(self, y):
        y_ = column_or_1d(y, warn=True)
        check_classification_targets(y)
        cls, y = np.unique(y_, return_inverse=True)
        self.class_weight_ = compute_class_weight(self.class_weight, cls, y_)
        if len(cls) < 2:
            raise ValueError(
                "The number of classes has to be greater than one; got %d"
                " class" % len(cls))

        self.classes_ = cls

        return np.asarray(y, dtype=np.float64, order='C')
    
    def _dense_fit(self, X, y, sample_weight, solver_type, kernel,
                   random_seed):
        #Precomputed
        if callable(self.kernel):
            self.__Xfit = X
            kernel = self.kernel(X, self.__Xfit)
            if sp.issparse(kernel):
                kernel = kernel.toarray()
            X = np.asarray(kernel, dtype=np.float64, order='C')

            if X.shape[0] != X.shape[1]:
                raise ValueError("X.shape[0] should be equal to X.shape[1]")
            kernel = "precomputed"

        #libsvm.set_verbosity_wrap(self.verbose)
        
        self.support_, self.support_vectors_, self.n_support_, \
            self.dual_coef_, self.intercept_, self.probA_, \
            self.probB_, self.fit_status_ = libsvm.fit(
                X, y,
                svm_type=solver_type, sample_weight=sample_weight,
                class_weight=self.class_weight_, kernel=kernel, C=self.C,
                nu=self.nu, probability=self.probability, degree=self.degree,
                shrinking=self.shrinking, tol=self.tol,
                cache_size=self.cache_size, coef0=self.coef0,
                gamma=self._gamma, epsilon=self.epsilon,
                max_iter=self.max_iter, random_seed=random_seed)

        if self.fit_status_ == 1:
            warnings.warn('Solver terminated early (max_iter=%i).'
                          '  Consider pre-processing your data with'
                          ' StandardScaler or MinMaxScaler.'
                          % self.max_iter, RuntimeWarning)


    def _sparse_fit(self, X, y, sample_weight, solver_type, kernel,
                    random_seed):
        X.data = np.asarray(X.data, dtype=np.float64, order='C')
        X.sort_indices()

        kernel_type = self._sparse_kernels.index(kernel)

        libsvm_sparse.set_verbosity_wrap(self.verbose)

        self.support_, self.support_vectors_, dual_coef_data, \
            self.intercept_, self.n_support_, \
            self.probA_, self.probB_, self.fit_status_ = \
            libsvm_sparse.libsvm_sparse_train(
                X.shape[1], X.data, X.indices, X.indptr, y, solver_type,
                kernel_type, self.degree, self._gamma, self.coef0, self.tol,
                self.C, self.class_weight_,
                sample_weight, self.nu, self.cache_size, self.epsilon,
                int(self.shrinking), int(self.probability), self.max_iter,
                random_seed)

        if self.fit_status_ == 1:
            warnings.warn('Solver terminated early (max_iter=%i).'
                          '  Consider pre-processing your data with'
                          ' StandardScaler or MinMaxScaler.'
                          % self.max_iter, Warning)

        if hasattr(self, "classes_"):
            n_class = len(self.classes_) - 1
        else:  # regression
            n_class = 1
        n_SV = self.support_vectors_.shape[0]

        dual_coef_indices = np.tile(np.arange(n_SV), n_class)
        dual_coef_indptr = np.arange(0, dual_coef_indices.size + 1,
                                     dual_coef_indices.size / n_class)
        self.dual_coef_ = sp.csr_matrix(
            (dual_coef_data, dual_coef_indices, dual_coef_indptr),
            (n_class, n_SV))
    
    def predict(self, X):
        X = self._validate_for_predict(X)
        predict = self._sparse_predict if self._sparse else self._dense_predict
        return predict(X)
    
    def _dense_predict(self, X):
        kernel = self.kernel
        if callable(self.kernel):
            if X.shape[1] != self.shape_fit_[0]:
                raise ValueError("X.shape[1] = %d should be equal to %d, "
                                 "the number of samples at training time" %
                                 (X.shape[1], self.shape_fit_[0]))
            kernel = self.kernel(X, self.__Xfit)
            if sp.issparse(kernel):
                kernel = kernel.toarray()
            kernel = 'precomputed'
            X = np.asarray(kernel, dtype=np.float64, order='C')

        if X.ndim == 1:
            X = check_array(X, order='C', accept_large_sparse=False)


        svm_type = 0 #for svc

        return libsvm.predict(
            X, self.support_, self.support_vectors_, self.n_support_,
            self._dual_coef_, self._intercept_,
            self.probA_, self.probB_, svm_type=svm_type, kernel=kernel,
            degree=self.degree, coef0=self.coef0, gamma=self._gamma,
            cache_size=self.cache_size)

    def _sparse_predict(self, X):
        # Precondition: X is a csr_matrix of dtype np.float64.
        kernel = self.kernel
        if callable(kernel):
            kernel = 'precomputed'

        kernel_type = self._sparse_kernels.index(kernel)

        C = 0.0  # C is not useful here

        return libsvm_sparse.libsvm_sparse_predict(
            X.data, X.indices, X.indptr,
            self.support_vectors_.data,
            self.support_vectors_.indices,
            self.support_vectors_.indptr,
            self._dual_coef_.data, self._intercept_,
            0, kernel_type,
            self.degree, self._gamma, self.coef0, self.tol,
            C, self.class_weight_,
            self.nu, self.epsilon, self.shrinking,
            self.probability, self.n_support_,
            self.probA_, self.probB_)


    def _decision_function(self, X):

        X = self._validate_for_predict(X)
        if callable(self.kernel):
            # in the case of precomputed kernel given as a function, we
            # have to compute explicitly the kernel matrix
            kernel = self.kernel(X, self.__Xfit)
            if sp.issparse(kernel):
                kernel = kernel.toarray()
            X = np.asarray(kernel, dtype=np.float64, order='C')

        if self._sparse:
            dec_func = self._sparse_decision_function(X)
        else:
            dec_func = self._dense_decision_function(X)

        # In binary case, we need to flip the sign of coef, intercept and
        # decision function.
        if len(self.classes_) == 2:
            return -dec_func.ravel()

        return dec_func
    

    def _dense_decision_function(self, X):
        X = check_array(X, dtype=np.float64, order="C",
                        accept_large_sparse=False)

        kernel = self.kernel
        if callable(kernel):
            kernel = 'precomputed'

        return libsvm.decision_function(
            X, self.support_, self.support_vectors_, self.n_support_,
            self._dual_coef_, self._intercept_,
            self.probA_, self.probB_,
            svm_type=0,
            kernel=kernel, degree=self.degree, cache_size=self.cache_size,
            coef0=self.coef0, gamma=self._gamma)
    

    def _sparse_decision_function(self, X):
        X.data = np.asarray(X.data, dtype=np.float64, order='C')

        kernel = self.kernel
        if callable(kernel):
            kernel = 'precomputed'

        kernel_type = self._sparse_kernels.index(kernel)

        return libsvm_sparse.libsvm_sparse_decision_function(
            X.data, X.indices, X.indptr,
            self.support_vectors_.data,
            self.support_vectors_.indices,
            self.support_vectors_.indptr,
            self._dual_coef_.data, self._intercept_,
            0, kernel_type,
            self.degree, self._gamma, self.coef0, self.tol,
            self.C, self.class_weight_,
            self.nu, self.epsilon, self.shrinking,
            self.probability, self.n_support_,
            self.probA_, self.probB_)
    

    def _validate_for_predict(self, X):
        check_is_fitted(self, 'support_')

        X = check_array(X, accept_sparse='csr', dtype=np.float64, order="C",
                        accept_large_sparse=False)
        if self._sparse and not sp.isspmatrix(X):
            X = sp.csr_matrix(X)
        if self._sparse:
            X.sort_indices()

        if sp.issparse(X) and not self._sparse and not callable(self.kernel):
            raise ValueError(
                "cannot use sparse input in %r trained on dense data"
                % type(self).__name__)
        n_samples, n_features = X.shape

        if self.kernel == "precomputed":
            if X.shape[1] != self.shape_fit_[0]:
                raise ValueError("X.shape[1] = %d should be equal to %d, "
                                 "the number of samples at training time" %
                                 (X.shape[1], self.shape_fit_[0]))
        elif n_features != self.shape_fit_[1]:
            raise ValueError("X.shape[1] = %d should be equal to %d, "
                             "the number of features at training time" %
                             (n_features, self.shape_fit_[1]))
        return X


    @property
    def coef_(self):
        if self.kernel != 'linear':
            raise AttributeError('coef_ is only available when using a '
                                 'linear kernel')

        coef = self._get_coef()

        # coef_ being a read-only property, it's better to mark the value as
        # immutable to avoid hiding potential bugs for the unsuspecting user.
        if sp.issparse(coef):
            # sparse matrix do not have global flags
            coef.data.flags.writeable = False
        else:
            # regular dense array
            coef.flags.writeable = False
        return coef

    def _get_coef(self):
        return safe_sparse_dot(self._dual_coef_, self.support_vectors_)

#Temporary function

def print_dict(my_dict):
    print("Data:")
    [print(i, ": ", my_dict[i]) for i in my_dict]
    
if __name__ == "__main__":
    print(libsvm.set_verbosity_wrap, libsvm.fit)
