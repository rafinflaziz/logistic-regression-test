import numpy as np

class LogisticRegression():

    def __init__(self):
        """Initialize model weights, number of samples, and number of features"""
        self.weights = None
        self.m = 0
        self.n = 0

    def __sigmoid(self, X):
        "Return sigmoid function with current weights evaluated at X"
        z = (X @ self.weights)
        return 1/(1 + np.exp(-z))

    def __loss_function(self, h, y):
        """
        Compute the binary cross-entropy (log loss) between predicted probabilities and true labels.

        Parameters:
        - h (numpy.ndarray): Predicted probabilities (float) for each sample.
        - y (numpy.ndarray): True labels (binary, 0 or 1) for each sample.

        Returns:
        float: The mean binary cross-entropy (log loss) over all samples.
        """
        return np.mean((-y * np.log(h)) - ((1-y) * np.log(1-h)))

    def __gradient_descent(self, X, y, alpha, n_iter):
        """
        Perform logistic regression parameter optimization using gradient descent.

        Parameters:
        - X (numpy.ndarray): Feature matrix of shape (m, n), where m is the number of samples
          and n is the number of features.
        - y (numpy.ndarray): True labels (binary, 0 or 1) for each sample.
        - alpha (float): Learning rate, determining the step size in each iteration.
        - n_iter (int): Number of iterations for gradient descent.
        """
        for i in range(n_iter):
            h = self.__sigmoid(X)
            self.weights = self.weights - (alpha * (X.T @ (h - y) / self.m))

    def fit(self, X, y, alpha=0.1, n_iter=1000):
        """
        Fit the logistic regression model to the given training data using gradient descent.

        Parameters:
        - X (numpy.ndarray): Feature matrix of shape (m, n), where m is the number of samples
          and n is the number of features.
        - y (pandas.Series or numpy.ndarray): True labels (binary, 0 or 1) for each sample.
        - alpha (float, optional): Learning rate, determining the step size in each iteration. Default is 0.1.
        - n_iter (int, optional): Number of iterations for gradient descent. Default is 1000.

        Returns:
        numpy.ndarray: Optimized weights for the logistic regression model.
        """
        self.m, self.n = X.shape
        
        # Adding a dummy feature for the bias weight
        X = np.hstack((np.ones((self.m, 1)), X))
        
        # Reshaping labels
        y = y.to_numpy().reshape(self.m, 1)
        
        # Initialize weights and do gradient descent
        self.weights = np.zeros(shape=(self.n + 1, 1))
        self.__gradient_descent(X, y, alpha, n_iter)

        return self.weights

    def predict(self, X):
        """
        Predict binary labels for given input data using the trained logistic regression model.
        The resulting probabilities are rounded to obtain binary predictions.
        """
        # Adding a dummy feature for the bias weight
        X = X.to_numpy()
        X = np.hstack((np.ones((self.m, 1)), X))

        res = self.__sigmoid(X)
        return np.round(res)