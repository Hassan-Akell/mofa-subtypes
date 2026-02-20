import numpy as np
import scipy.special as sp
from mofapy2.core.distributions.basic_distributions import Distribution

class NormalGamma(Distribution):
    """
    Normal–Gamma variational distribution:
        q(mu, lambda) = N(mu | m, (t*lambda)^-1) * Gamma(lambda | a, b)
    with shape-rate parameterization for Gamma.

    Stored parameters: m, t, a, b.

    Provides expectations needed by DP-Cluster-MOFA updates:
      - E_lambda, E_log_lambda
      - E_lambda_mu, E_lambda_mu2
    """

    def __init__(self, dim, m, t, a, b):
        super().__init__(dim)
        self.params = {
            "m": np.asarray(m, dtype=float),
            "t": np.asarray(t, dtype=float),
            "a": np.asarray(a, dtype=float),
            "b": np.asarray(b, dtype=float),
        }
        self.expectations = {}
        self.updateExpectations()

    def getParameters(self):
        return self.params

    def setParameters(self, **kwargs):
        for k in ["m", "t", "a", "b"]:
            if k in kwargs:
                self.params[k] = np.asarray(kwargs[k], dtype=float)

    def updateExpectations(self):
        m = self.params["m"]
        t = np.clip(self.params["t"], 1e-30, np.inf)
        a = np.clip(self.params["a"], 1e-30, np.inf)
        b = np.clip(self.params["b"], 1e-30, np.inf)

        E_lambda = a / b
        E_log_lambda = sp.digamma(a) - np.log(b)
        E_lambda_mu = m * E_lambda
        E_lambda_mu2 = (m**2) * E_lambda + (1.0 / t)

        self.expectations = {
            "E_lambda": E_lambda,
            "E_log_lambda": E_log_lambda,
            "E_lambda_mu": E_lambda_mu,
            "E_lambda_mu2": E_lambda_mu2,
            "E_mu": m,
        }

    def getExpectations(self):
        return self.expectations
