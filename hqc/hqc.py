import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.preprocessing import normalize
from joblib import Parallel, delayed

class HQC(BaseEstimator, ClassifierMixin):
    """The Helstrom Quantum Centroid (HQC) classifier is a quantum-inspired supervised 
    classification approach for data with binary classes (ie. data with 2 classes only).
                         
    Parameters
    ----------
    rescale : int or float, default = 1
        The dataset rescaling factor. A parameter used for rescaling the dataset. 
    encoding : str, default = 'amplit'
        The encoding method used to encode vectors into quantum densities. Possible values:
        'amplit', 'stereo'. 'amplit' means using the amplitude encoding method. 'stereo' means 
        using the inverse of the standard stereographic projection encoding method. Default set 
        to 'amplit'.
    n_copies : int, default = 1
        The number of copies to take for each quantum density. This is equivalent to taking 
        the n-fold Kronecker tensor product for each quantum density.
    class_wgt : str, default = 'equi'
        The class weights assigned to the Quantum Helstrom observable terms. Possible values: 
        'equi', 'weighted'. 'equi' means assigning equal weights of 1/2 (equiprobable) to the
        two classes in the Quantum Helstrom observable. 'weighted' means assigning weights equal 
        to the proportion of the number of rows in each class to the two classes in the Quantum 
        Helstrom observable. Default set to 'equi'.
    n_jobs : int, default = None
        The number of CPU cores used when parallelizing. If -1 all CPUs are used. If 1 is given, 
        no parallel computing code is used at all. For n_jobs below -1, (n_cpus + 1 + n_jobs) 
        are used. Thus for n_jobs = -2, all CPUs but one are used. None is a marker for ‘unset’ 
        that will be interpreted as n_jobs = 1.
    n_splits : int, default = 1
        The number of subset splits performed on the input dataset row-wise and on the number 
        of eigenvalues/eigenvectors of the Quantum Helstrom observable for optimal speed 
        performance. If 1 is given, no splits are performed. For optimal speed, recommend using 
        n_splits = int(numpy.ceil(number of CPU cores used/2)). If memory blow-out occurs, 
        reduce n_splits.
    
    Attributes
    ----------
    classes_ : ndarray, shape (2,)
        Sorted binary classes.
    centroids_ : ndarray, shape (2, (n_features + 1)**n_copies, (n_features + 1)**n_copies)
        Quantum Centroids for class with index 0 and 1 respectively.
    hels_obs_ : ndarray, shape ((n_features + 1)**n_copies, (n_features + 1)**n_copies)
        Quantum Helstrom observable.
    proj_sums_ : tuple, shape (2, (n_features + 1)**n_copies, (n_features + 1)**n_copies)
        Sum of the projectors of the Quantum Helstrom observable's eigenvectors, which has
        corresponding positive and negative eigenvalues respectively.
    hels_bound_ : float
        Helstrom bound is the upper bound of the probability that one can correctly 
        discriminate whether a quantum density is of which of the two binary quantum density 
        pattern.          
    """
    # Added binary_only tag as required by sklearn check_estimator
    def _more_tags(self):
        return {'binary_only': True}
    
    
    # Initialize model hyperparameters
    def __init__(self, 
                 rescale = 1,
                 encoding = 'amplit',
                 n_copies = 1, 
                 class_wgt = 'equi', 
                 n_jobs = None, 
                 n_splits = 1):
        self.rescale = rescale
        self.encoding = encoding
        self.n_copies = n_copies
        self.class_wgt = class_wgt
        self.n_jobs = n_jobs
        self.n_splits = n_splits
        
    
    # Function for fit
    def fit(self, X, y):
        """Perform HQC classification with the inverse of the standard stereographic 
        projection encoding, with the option to rescale the dataset prior to encoding.
                
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples. An array of int or float.
        y : array-like, shape (n_samples,)
            The training input binary target values. An array of str, int or float.
            
        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Ensure target y is of non-regression type  
        # Added as required by sklearn check_estimator
        check_classification_targets(y)
    
        # Store binary classes and encode y into binary class indexes 0 and 1
        self.classes_, y_class_index = np.unique(y, return_inverse = True)
        
        # Raise error if there are more than 2 classes
        if len(self.classes_) > 2:
            raise ValueError('only 2 classes are supported')
        
        # Cast X to float to ensure all following calculations below are done in float  
        # rather than integer
        X = X.astype(float)
        
        # Rescale X
        X = self.rescale*X
        
        # Calculate sum of squares of each row (sample) in X
        X_sq_sum = (X**2).sum(axis = 1)
        
        # Number of rows in X
        m = X.shape[0]
        
        # Number of columns in X
        n = X.shape[1]
        
        # Calculate X' using amplitude or inverse of the standard stereographic projection 
        # encoding method
        if self.encoding == 'amplit':
            X_prime = normalize(np.concatenate((X, np.ones(m).reshape(-1, 1)), axis = 1))
        elif self.encoding == 'stereo':
            X_prime = (1 / (X_sq_sum + 1)).reshape(-1, 1) \
                      *(np.concatenate((2*X, (X_sq_sum - 1).reshape(-1, 1)), axis = 1))
        else:
            raise ValueError('encoding should be "amplit" or "stereo"')
            
        # Function to calculate terms in the Quantum Centroids and quantum Helstrom 
        # observable for each class
        def centroids_terms_func(i):
            # Determine rows (samples) in X' belonging to either class
            X_prime_class = X_prime[y_class_index == i]
            
            # Number of rows (samples) in X' belonging to either class
            m_class = X_prime_class.shape[0]
            
            # Split X' belonging to either class into n_splits subsets, row-wise
            X_prime_class_split = np.array_split(X_prime_class, 
                                                 indices_or_sections = self.n_splits, 
                                                 axis = 0)
            
            # Function to calculate terms in the Quantum Centroids and quantum Helstrom
            # observable for each class, per subset split
            def X_prime_class_split_func(j):
                # Counter for j-th split of X'
                X_prime_class_split_jth = X_prime_class_split[j]
                
                # Number of rows (samples) in j-th split of X'
                m_class_split = X_prime_class_split_jth.shape[0]
            
                # Number of rows/columns in density matrix
                density_nrow_ncol = (n + 1)**self.n_copies
            
                # Initialize arrays density_sum, centroid and hels_obs_terms
                density_sum = np.zeros((density_nrow_ncol, density_nrow_ncol))
                centroid = density_sum
                hels_obs_terms = density_sum
                for k in range(m_class_split):
                    # Encode vectors into quantum densities
                    X_prime_class_split_each_row = X_prime_class_split_jth[k, :]
                    density_each_row = np.dot(X_prime_class_split_each_row.reshape(-1, 1),
                                              X_prime_class_split_each_row.reshape(1, -1))
                
                    # Calculate n-fold Kronecker tensor product
                    if self.n_copies == 1:
                        density_each_row = density_each_row
                    else:
                        density_each_row_copy = density_each_row
                        for u in range(self.n_copies - 1):
                            density_each_row = np.kron(density_each_row, density_each_row_copy)
                
                    # Calculate sum of quantum densities
                    density_sum = density_sum + density_each_row
                
                    # Calculate Quantum Centroid
                    # Added ZeroDivisionError as required by sklearn check_estimator
                    try:
                        centroid = (1 / m_class)*density_sum
                    except ZeroDivisionError:
                        centroid = 0
                    
                    # Calculate terms in the quantum Helstrom observable
                    if self.class_wgt == 'equi':
                        hels_obs_terms = 0.5*centroid
                    elif self.class_wgt == 'weighted':
                        hels_obs_terms = (1 / m)*density_sum
                    else:
                        raise ValueError('class_wgt should be "equi" or "weighted"')                    
                return m_class_split, centroid, hels_obs_terms        
            return np.sum(Parallel(n_jobs = self.n_jobs) \
                         (delayed(X_prime_class_split_func)(j) for j in range(self.n_splits)), axis = 0)
            
        # Calculate Quantum Centroids and terms in the quantum Helstrom observable for each class
        centroids_terms = np.array(Parallel(n_jobs = self.n_jobs) \
                                          (delayed(centroids_terms_func)(i) for i in range(2)))
        
        # Determine Quantum Centroids
        self.centroids_ = centroids_terms[:, 1]
           
        # Calculate quantum Helstrom observable
        self.hels_obs_ = centroids_terms[0, 2] - centroids_terms[1, 2]     
        
        # Calculate eigenvalues w and eigenvectors v of the quantum Helstrom observable
        w, v = np.linalg.eigh(self.hels_obs_)
        
        # Length of w
        len_w = len(w)
        
        # Initialize array eigval_class
        eigval_class = np.empty_like(w)
        for i in range(len_w):
            # Create an array of 0s and 1s to indicate positive and negative eigenvalues
            # respectively
            if w[i] > 0:
                eigval_class[i] = 0
            else:
                eigval_class[i] = 1
        
        # Transpose matrix v containing eigenvectors to row-wise
        eigvec = v.T
        
        # Function to calculate sum of the projectors corresponding to positive and negative
        # eigenvalues respectively
        def sum_proj_func(i):
            # Determine eigenvectors belonging to positive and negative eigenvalues respectively
            eigvec_class = eigvec[eigval_class == i]
            
            # Split eigenvectors into n_splits subsets
            eigvec_class_split = np.array_split(eigvec_class, 
                                                indices_or_sections = self.n_splits, 
                                                axis = 0)
            
            # Function to calculate sum of the projectors corresponding to positive and negative
            # eigenvalues respectively, per subset split
            def eigvec_class_split_func(j):
                # Initialize array proj_sums_split
                proj_sums_split = np.zeros_like(self.hels_obs_)
                for k in eigvec_class_split[j]:
                    # Calculate sum of the projectors corresponding to positive and negative
                    # eigenvalues respectively, per subset split
                    proj_sums_split = proj_sums_split + np.dot(k.reshape(-1, 1), k.reshape(1, -1))
                return proj_sums_split        
            return np.sum(Parallel(n_jobs = self.n_jobs) \
                         (delayed(eigvec_class_split_func)(j) for j in range(self.n_splits)), axis = 0)
        
        # Calculate sum of the projectors corresponding to positive and negative eigenvalues
        # respectively
        self.proj_sums_ = Parallel(n_jobs = self.n_jobs) \
                          (delayed(sum_proj_func)(i) for i in range(2))    
                       
        # Calculate Helstrom bound
        self.hels_bound_ = (centroids_terms[0, 0] / m)*np.einsum('ij,ji->', self.centroids_[0], 
                                                                self.proj_sums_[0]) \
                           + (centroids_terms[1, 0] / m)*np.einsum('ij,ji->', self.centroids_[1], 
                                                                  self.proj_sums_[1])
        return self
        
    
    # Function for predict_proba
    def predict_proba(self, X):
        """Performs HQC classification on X and returns the trace of the dot product of the densities 
        and the sum of the projectors with corresponding positive and negative eigenvalues respectively.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples. An array of int or float.       
            
        Returns
        -------
        trace_matrix : array-like, shape (n_samples, 2)
            Column index 0 corresponds to the trace of the dot product of the densities and the sum  
            of projectors with positive eigenvalues. Column index 1 corresponds to the trace of the  
            dot product of the densities and the sum of projectors with negative eigenvalues. An array 
            of float.
        """
        # Check if fit had been called
        check_is_fitted(self, ['proj_sums_'])

        # Input validation
        X = check_array(X)
        
        # Cast X to float to ensure all following calculations below are done in float 
        # rather than integer
        X = X.astype(float)        
        
        # Rescale X
        X = self.rescale*X        
        
        # Calculate sum of squares of each row (sample) in X
        X_sq_sum = (X**2).sum(axis = 1)
        
        # Number of rows in X
        m = X.shape[0]
        
        # Number of columns in X
        n = X.shape[1]

        # Calculate X' using amplitude or inverse of the standard stereographic projection 
        # encoding method
        if self.encoding == 'amplit':
            X_prime = normalize(np.concatenate((X, np.ones(m).reshape(-1, 1)), axis = 1))
        elif self.encoding == 'stereo':
            X_prime = (1 / (X_sq_sum + 1)).reshape(-1, 1) \
                      *(np.concatenate((2*X, (X_sq_sum - 1).reshape(-1, 1)), axis = 1))
        else:
            raise ValueError('encoding should be "amplit" or "stereo"')
               
        # Function to calculate trace values for each class
        def trace_func(i):
            # Split X' into n_splits subsets, row-wise
            X_prime_split = np.array_split(X_prime, 
                                           indices_or_sections = self.n_splits, 
                                           axis = 0)
            
            # Function to calculate trace values for each class, per subset split
            def trace_split_func(j):
                # Counter for j-th split X'
                X_prime_split_jth = X_prime_split[j]
                
                # Number of rows (samples) in j-th split X'
                X_prime_split_m = X_prime_split_jth.shape[0]
                
                # Initialize array trace_class_split
                trace_class_split = np.empty(X_prime_split_m)
                for k in range(X_prime_split_m):
                    # Encode vectors into quantum densities
                    X_prime_split_each_row = X_prime_split_jth[k, :]
                    density_each_row = np.dot(X_prime_split_each_row.reshape(-1, 1), 
                                              X_prime_split_each_row.reshape(1, -1))
                
                    # Calculate n-fold Kronecker tensor product
                    if self.n_copies == 1:     
                        density_each_row = density_each_row
                    else:
                        density_each_row_copy = density_each_row
                        for u in range(self.n_copies - 1):
                            density_each_row = np.kron(density_each_row, density_each_row_copy)
                        
                    # Calculate trace of the dot product of density of each row and sum of projectors 
                    # with corresponding positive and negative eigenvalues respectively    
                    trace_class_split[k] = np.einsum('ij,ji->', density_each_row, self.proj_sums_[i])
                return trace_class_split
            
            # Calculate trace values for each class, per subset split
            trace_class = Parallel(n_jobs = self.n_jobs) \
                          (delayed(trace_split_func)(j) for j in range(self.n_splits))
            return np.concatenate(trace_class, axis = 0)
            
        # Calculate trace values for each class
        trace_matrix = np.transpose(Parallel(n_jobs = self.n_jobs) \
                                   (delayed(trace_func)(i) for i in range(2)))
        return trace_matrix
        
    
    # Function for predict
    def predict(self, X):
        """Performs HQC classification on X and returns the binary classes.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples. An array of int or float.
            
        Returns
        -------
        self.classes_[predict_trace_index] : array-like, shape (n_samples,)
            The predicted binary classes. An array of str, int or float.
        """
        # Determine column index with the higher trace value in trace_matrix
        # If both columns have the same trace value, returns column index 0
        predict_trace_index = np.argmax(self.predict_proba(X), axis = 1)
        # Returns the predicted binary classes
        return self.classes_[predict_trace_index]
