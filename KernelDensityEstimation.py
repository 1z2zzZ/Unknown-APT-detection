import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV


class KernelDensityEstimation:
    def __init__(self, kernel="gaussian", bandwidth="auto"):
        self.kernel = kernel
        self.bandwidth = bandwidth

    def fit(self, x):
        self.x = x
        if self.bandwidth == "auto":
            self.__cross_validation()
        self.kde = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth)
        self.kde.fit(self.x)

    def predict(self, x):
        # Calculate probability density estimation
        probability_density = np.exp(self.kde.score_samples(x))
        return probability_density

    def __cross_validation(self):
        # Cross-validation to choose the optimal bandwidth
        bandwidths = np.logspace(-1, 1, 20)
        grid = GridSearchCV(KernelDensity(kernel=self.kernel), {'bandwidth': bandwidths}, cv=5)
        grid.fit(self.x)
        self.bandwidth = grid.best_params_['bandwidth']
