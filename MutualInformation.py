import numpy as np

class MutualInformation:
    def __init__(self, kernel="gaussian", bandwidth=1.0, MinDouble=1e-15):
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.MinDouble = MinDouble

    def __union_probability(self, x, y):
        # Calculate the joint probability density function
        p_xy = KernelDensityEstimation(kernel=self.kernel, bandwidth=self.bandwidth)
        p_xy.fit(np.hstack((x, y)))
        return p_xy

    def __marginal_probability(self, x, y):
        # Calculate the marginal probability density function
        p_x = KernelDensityEstimation(kernel=self.kernel, bandwidth=self.bandwidth)
        p_x.fit(x)
        p_y = KernelDensityEstimation(kernel=self.kernel, bandwidth=self.bandwidth)
        p_y.fit(y)
        return p_x, p_y

    def __mutual_information(self, x, y):
        # Calculate mutual information
        p_xy = self.__union_probability(x, y)
        p_x, p_y = self.__marginal_probability(x, y)
        # Use multiple integrals to calculate mutual information
        mi = 0
        h_x = 0
        h_y = 0
        for i in range(len(x)):
            for j in range(len(y)):
                p_xy_prob = p_xy.predict(np.array([[x[i][0], y[j][0]]])) + self.MinDouble
                p_x_prob = p_x.predict(np.array([x[i]])) + self.MinDouble
                p_y_prob = p_y.predict(np.array([y[j]])) + self.MinDouble
                mi += p_xy_prob * np.log2(p_xy_prob / (p_x_prob * p_y_prob))
        for i in range(len(x)):
            p_x_prob = p_x.predict(np.array([x[i]])) + self.MinDouble
            h_x += -p_x_prob * np.log2(p_x_prob)
        for j in range(len(y)):
            p_y_prob = p_y.predict(np.array([y[j]])) + self.MinDouble
            h_y += -p_y_prob * np.log2(p_y_prob)
        # Normalize
        mi = 2*mi / (h_x + h_y)

        return mi

    def fit_transform(self, x, y):
        x, y = np.array(x), np.array(y)
        # Check the validity of x and y
        self.__check_validity(x, y)
        # Convert x and y to (-1, 1) dimensional arrays
        X = x.reshape(-1, 1)
        Y = y.reshape(-1, 1)
        # Calculate mutual information
        mi = self.__mutual_information(X, Y)
        return mi

    def __check_validity(self, x, y):
        # Check if the dimensions of x and y are the same
        if x.ndim != y.ndim:
            raise ValueError("The dimensions of x and y are different")
        # Check if the lengths of x and y are the same
        if len(x) != len(y):
            raise ValueError("The lengths of x and y are different")
        # Check if the dimensions of x and y are 1
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("The dimensions of x and y should be 1")
