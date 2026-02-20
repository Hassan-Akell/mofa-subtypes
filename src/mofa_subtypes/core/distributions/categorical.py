import numpy as np
from mofapy2.core.distributions.basic_distributions import Distribution

class Categorical(Distribution):
    """
    Simple categorical distribution for VI.

    Parameters
    ----------
    theta : array-like, shape (..., C)
        Probabilities that sum to 1 on the last axis.
    """

    def __init__(self, dim, theta, lnE=None):
        super().__init__(dim)
        self.params = {"theta": np.asarray(theta, dtype=float)}
        self.expectations = {}
        if lnE is None:
            self.updateExpectations()
        else:
            self.expectations["E"] = self.params["theta"]
            self.expectations["lnE"] = lnE

    def updateExpectations(self):
        theta = self.params["theta"]
        eps = 1e-30
        self.expectations["E"] = theta
        self.expectations["lnE"] = np.log(np.clip(theta, eps, 1.0))

    def getParameters(self):
        return self.params

    def setParameters(self, **kwargs):
        if "theta" in kwargs:
            self.params["theta"] = np.asarray(kwargs["theta"], dtype=float)

    def getExpectations(self):
        return self.expectations
