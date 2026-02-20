from __future__ import annotations

import numpy as np

from mofapy2.core.nodes.variational_nodes import (
    Variational_Node,
    Unobserved_Variational_Node,
    Gamma_Unobserved_Variational_Node,
    Beta_Unobserved_Variational_Node,
    UnivariateGaussian_Unobserved_Variational_Node,
)
from mofapy2.core.nodes.Z_nodes import Z_Node

from ..distributions import Categorical, NormalGamma


def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    x = logits - np.max(logits, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


class Z_DP_Node(Z_Node):
    """Z node with an overridden ELBO contribution for the DP-mixture prior."""

    def calculateELBO(self):
        # Start with the standard entropy term of q(z)
        entropy_qz = self.Q.entropy()

        # If DP blocks aren't connected, fall back to base behaviour
        if not ("C" in self.markov_blanket and "MuLambda" in self.markov_blanket):
            return -entropy_qz

        R = self.markov_blanket["C"].getExpectations()["E"]  # (N, C)
        ml = self.markov_blanket["MuLambda"].getExpectations()
        Eloglambda = ml["E_log_lambda"]  # (C, K)
        Elambda = ml["E_lambda"]         # (C, K)
        Elambda_mu = ml["E_lambda_mu"]   # (C, K)
        Elambda_mu2 = ml["E_lambda_mu2"] # (C, K)

        Ez = self.Q.getExpectations()["E"]   # (N, K)
        Ez2 = self.Q.getExpectations()["E2"] # (N, K)

        # Expected log p(z | c, mu, lambda) under q
        # sum_{n,c,k} r_nc * 0.5*( Eloglambda_ck - log(2pi) - Elambda_ck*Ez2_nk + 2*Elambda_mu_ck*Ez_nk - Elambda_mu2_ck )
        const_ck = 0.5 * (Eloglambda - np.log(2.0 * np.pi) - Elambda_mu2)  # (C, K)
        term_const = (R @ np.sum(const_ck, axis=1))  # (N,)  [sum_k inside]
        # Quadratic and linear terms via matrix multiplications
        term_quad = -0.5 * (Ez2 @ Elambda.T)         # (N, C)
        term_lin  = (Ez @ Elambda_mu.T)              # (N, C)

        # combine: sum_c r_nc [sum_k const_ck + term_quad + term_lin]
        lp = np.sum(R * (term_quad + term_lin), axis=1) + term_const

        return float(np.sum(lp) - entropy_qz)


class Responsibilities_Node(Unobserved_Variational_Node):
    """Responsibilities q(c_n) = Cat(r_n), n=1..N."""

    def __init__(self, N: int, C: int, init: str = "uniform", seed: int | None = None):
        super().__init__(dim=(N, C))
        rng = np.random.default_rng(seed)
        if init == "random":
            theta = rng.random((N, C))
            theta = theta / theta.sum(axis=1, keepdims=True)
        else:
            theta = np.ones((N, C)) / C
        self.Q = Categorical(dim=(N, C), theta=theta)

    def updateParameters(self, ix=None, ro=None):
        # No mini-batching for C in this initial implementation.
        assert "V" in self.markov_blanket and "Z" in self.markov_blanket and "MuLambda" in self.markov_blanket

        V = self.markov_blanket["V"].Q  # Beta distribution from mofapy2
        ml = self.markov_blanket["MuLambda"].getExpectations()
        Z = self.markov_blanket["Z"].Q.getExpectations()

        Ez = Z["E"]   # (N, K)
        Ez2 = Z["E2"] # (N, K)

        Eloglambda = ml["E_log_lambda"]   # (C, K)
        Elambda = ml["E_lambda"]          # (C, K)
        Elambda_mu = ml["E_lambda_mu"]    # (C, K)
        Elambda_mu2 = ml["E_lambda_mu2"]  # (C, K)

        # E[log pi_c] from stick-breaking: pi_c = v_c * prod_{l<c}(1-v_l), with v_C = 1
        ln_v = V.getExpectations()["lnE"]        # (C-1,)
        ln_1mv = V.getExpectations()["lnEInv"]   # (C-1,)
        Cc = Elambda.shape[0]
        assert Cc >= 1
        Elogpi = np.zeros(Cc)
        cumsum_ln1mv = np.concatenate([[0.0], np.cumsum(ln_1mv)])  # length C
        for c in range(Cc):
            if c < Cc - 1:
                Elogpi[c] = ln_v[c] + cumsum_ln1mv[c]
            else:
                Elogpi[c] = cumsum_ln1mv[c]

        # compute log responsibilities
        # log r_nc ∝ Elogpi_c + 0.5 * sum_k( Eloglambda_ck - log(2pi) - Elambda_ck*Ez2_nk + 2Elambda_mu_ck*Ez_nk - Elambda_mu2_ck )
        const_c = 0.5 * (np.sum(Eloglambda - np.log(2.0 * np.pi) - Elambda_mu2, axis=1))  # (C,)
        term_quad = -0.5 * (Ez2 @ Elambda.T)       # (N, C)
        term_lin  = (Ez @ Elambda_mu.T)            # (N, C)
        log_r = Elogpi[None, :] + const_c[None, :] + term_quad + term_lin

        r = _softmax(log_r, axis=1)
        self.Q.setParameters(theta=r)
        self.Q.updateExpectations()

    def calculateELBO(self):
        # E_q[log p(c|v)] - E_q[log q(c)]
        if "V" not in self.markov_blanket:
            return 0.0
        V = self.markov_blanket["V"].Q
        R = self.Q.getExpectations()["E"]

        ln_v = V.getExpectations()["lnE"]
        ln_1mv = V.getExpectations()["lnEInv"]
        Cc = R.shape[1]
        Elogpi = np.zeros(Cc)
        cumsum_ln1mv = np.concatenate([[0.0], np.cumsum(ln_1mv)])
        for c in range(Cc):
            if c < Cc - 1:
                Elogpi[c] = ln_v[c] + cumsum_ln1mv[c]
            else:
                Elogpi[c] = cumsum_ln1mv[c]

        # expected log prior
        lp = np.sum(R * Elogpi[None, :])

        # entropy of categorical
        eps = 1e-30
        ent = -np.sum(R * np.log(np.clip(R, eps, 1.0)))

        return float(lp + ent)


class DPGamma_Node(Gamma_Unobserved_Variational_Node):
    """DP concentration gamma with conjugate Gamma update for Beta(1,gamma) sticks."""

    def updateParameters(self, ix=None, ro=None):
        assert "V" in self.markov_blanket
        V = self.markov_blanket["V"].Q
        ln_1mv = V.getExpectations()["lnEInv"]  # (C-1,)

        a0 = self.P.getParameters()["a"]
        b0 = self.P.getParameters()["b"]

        Cminus = ln_1mv.shape[0]
        qa = a0 + Cminus
        qb = b0 - np.sum(ln_1mv)  # ln_1mv is negative -> increases rate

        # Optional SVI blending
        if ro is not None:
            old = self.Q.getParameters()
            qa = (1 - ro) * old["a"] + ro * qa
            qb = (1 - ro) * old["b"] + ro * qb

        self.Q.setParameters(a=qa, b=qb)
        self.updateExpectations()


class DPSticks_Node(Beta_Unobserved_Variational_Node):
    """Stick-breaking variables v_1..v_{C-1}."""

    def updateParameters(self, ix=None, ro=None):
        assert "C" in self.markov_blanket and "Gamma" in self.markov_blanket
        R = self.markov_blanket["C"].getExpectations()["E"]  # (N, C)
        Egamma = float(self.markov_blanket["Gamma"].Q.getExpectations()["E"])

        N, Cc = R.shape
        assert self.dim[0] == Cc - 1

        Nk = R.sum(axis=0)  # length C
        # N_{>c} for c=1..C-1
        Ngt = np.array([Nk[c+1:].sum() for c in range(Cc-1)])

        qa = 1.0 + Nk[:-1]
        qb = Egamma + Ngt

        if ro is not None:
            old = self.Q.getParameters()
            qa = (1 - ro) * old["a"] + ro * qa
            qb = (1 - ro) * old["b"] + ro * qb

        self.Q.setParameters(a=qa, b=qb)
        self.updateExpectations()


class MuLambda_Node(Unobserved_Variational_Node):
    """Component parameters (mu_ck, lambda_ck) with Normal–Gamma variational family."""

    def __init__(self, C: int, K: int, a0: float = 1.0, b0: float = 1.0, seed: int | None = None):
        super().__init__(dim=(C, K))
        rng = np.random.default_rng(seed)

        # Initialise somewhat broadly
        m = rng.normal(0.0, 1.0, size=(C, K))
        t = np.ones((C, K))
        a = np.ones((C, K)) * (a0 + 0.5)
        b = np.ones((C, K)) * (b0 + 0.5)

        self.a0 = float(a0)
        self.b0 = float(b0)

        self.Q = NormalGamma(dim=(C, K), m=m, t=t, a=a, b=b)

    def updateParameters(self, ix=None, ro=None):
        assert all(k in self.markov_blanket for k in ["C", "Z", "m0", "t0"])
        R = self.markov_blanket["C"].getExpectations()["E"]  # (N, C)
        Zexp = self.markov_blanket["Z"].Q.getExpectations()
        Ez = Zexp["E"]    # (N, K)
        Ez2 = Zexp["E2"]  # (N, K)

        m0 = self.markov_blanket["m0"].Q.getParameters()["mean"]  # (K,)
        t0 = float(self.markov_blanket["t0"].Q.getExpectations()["E"])

        N, Cc = R.shape
        K = Ez.shape[1]

        Nk = R.sum(axis=0)  # (C,)
        sum_Ez = R.T @ Ez    # (C, K)
        sum_Ez2 = R.T @ Ez2  # (C, K)

        # Avoid divide-by-zero for empty components
        Nk_safe = np.clip(Nk, 1e-30, np.inf)[:, None]
        xbar = sum_Ez / Nk_safe

        t_ck = t0 + Nk[:, None]
        m_ck = (t0 * m0[None, :] + sum_Ez) / np.clip(t_ck, 1e-30, np.inf)

        a_ck = self.a0 + 0.5 * Nk[:, None]

        # Weighted within-component scatter: S = sum Ez2 - N * xbar^2
        S = sum_Ez2 - Nk[:, None] * (xbar**2)

        # Prior mean deviation term
        dev = (xbar - m0[None, :]) ** 2
        b_ck = self.b0 + 0.5 * (S + (t0 * Nk[:, None] / np.clip(t_ck, 1e-30, np.inf)) * dev)

        if ro is not None:
            old = self.Q.getParameters()
            m_ck = (1 - ro) * old["m"] + ro * m_ck
            t_ck = (1 - ro) * old["t"] + ro * t_ck
            a_ck = (1 - ro) * old["a"] + ro * a_ck
            b_ck = (1 - ro) * old["b"] + ro * b_ck

        self.Q.setParameters(m=m_ck, t=t_ck, a=a_ck, b=b_ck)
        self.Q.updateExpectations()

    def getExpectations(self):
        # Provide a MOFA-like interface for downstream nodes
        exp = self.Q.getExpectations()
        return {
            "E_lambda": exp["E_lambda"],
            "E_log_lambda": exp["E_log_lambda"],
            "E_lambda_mu": exp["E_lambda_mu"],
            "E_lambda_mu2": exp["E_lambda_mu2"],
        }


class BaseMean_Node(UnivariateGaussian_Unobserved_Variational_Node):
    """Base mean m0k (k=1..K), Gaussian variational."""

    def updateParameters(self, ix=None, ro=None):
        assert all(k in self.markov_blanket for k in ["MuLambda", "t0"])
        ml = self.markov_blanket["MuLambda"].Q.getExpectations()
        Elambda = ml["E_lambda"]  # (C, K)
        Elambda_mu = ml["E_lambda_mu"]  # (C, K)

        t0 = float(self.markov_blanket["t0"].Q.getExpectations()["E"])

        # Prior params
        prior_mean = self.P.getParameters()["mean"]  # (K,)
        prior_var = self.P.getParameters()["var"]    # (K,)
        prior_prec = 1.0 / np.clip(prior_var, 1e-30, np.inf)

        # Posterior precision and mean
        post_prec = prior_prec + t0 * np.sum(Elambda, axis=0)
        post_mean = (prior_prec * prior_mean + t0 * np.sum(Elambda_mu, axis=0)) / np.clip(post_prec, 1e-30, np.inf)
        post_var = 1.0 / np.clip(post_prec, 1e-30, np.inf)

        if ro is not None:
            old = self.Q.getParameters()
            post_mean = (1 - ro) * old["mean"] + ro * post_mean
            post_var = (1 - ro) * old["var"] + ro * post_var

        self.Q.setParameters(mean=post_mean, var=post_var)
        self.updateExpectations()


class T0_Node(Gamma_Unobserved_Variational_Node):
    """Relative precision t0 (scalar) with Gamma variational."""

    def updateParameters(self, ix=None, ro=None):
        assert all(k in self.markov_blanket for k in ["MuLambda", "m0"])
        ml = self.markov_blanket["MuLambda"].Q.getExpectations()
        Elambda = ml["E_lambda"]          # (C, K)
        Elambda_mu = ml["E_lambda_mu"]    # (C, K)
        Elambda_mu2 = ml["E_lambda_mu2"]  # (C, K)

        m0 = self.markov_blanket["m0"].Q.getParameters()["mean"]  # (K,)

        a0 = float(self.P.getParameters()["a"])
        b0 = float(self.P.getParameters()["b"])

        Cc, K = Elambda.shape
        qa = a0 + 0.5 * Cc * K

        # sum_{c,k} E[lambda (mu-m0)^2]
        m0_b = m0[None, :]
        quad = Elambda_mu2 - 2.0 * m0_b * Elambda_mu + (m0_b**2) * Elambda
        qb = b0 + 0.5 * np.sum(quad)

        if ro is not None:
            old = self.Q.getParameters()
            qa = (1 - ro) * old["a"] + ro * qa
            qb = (1 - ro) * old["b"] + ro * qb

        self.Q.setParameters(a=qa, b=qb)
        self.updateExpectations()


class _ZPriorNode(Variational_Node):
    """Deterministic node that exposes per-(n,k) Gaussian prior moments for Z updates."""

    def __init__(self, dim):
        super().__init__(dim)
        self._value = np.zeros(dim)
        self.mini_batch = None

    def define_mini_batch(self, ix):
        self.mini_batch = ix

    def get_mini_batch(self):
        if self.mini_batch is None:
            return self._value
        return self._value[self.mini_batch]

    def getExpectations(self):
        # mimic Constant_Node interface
        return {"E": self._value, "E2": self._value**2}

    def getExpectation(self):
        return self._value

    def updateExpectations(self, dist=None):
        return

    def updateParameters(self, ix=None, ro=None):
        raise NotImplementedError


class MuZ_Node(_ZPriorNode):
    def __init__(self, N: int, K: int):
        super().__init__((N, K))

    def updateParameters(self, ix=None, ro=None):
        assert all(k in self.markov_blanket for k in ["C", "MuLambda"])
        R = self.markov_blanket["C"].getExpectations()["E"]  # (N, C)
        ml = self.markov_blanket["MuLambda"].getExpectations()
        A = R @ ml["E_lambda"]       # (N, K)
        B = R @ ml["E_lambda_mu"]    # (N, K)
        self._value = B / np.clip(A, 1e-30, np.inf)


class AlphaZ_Node(_ZPriorNode):
    def __init__(self, N: int, K: int):
        super().__init__((N, K))

    def updateParameters(self, ix=None, ro=None):
        assert all(k in self.markov_blanket for k in ["C", "MuLambda"])
        R = self.markov_blanket["C"].getExpectations()["E"]  # (N, C)
        ml = self.markov_blanket["MuLambda"].getExpectations()
        A = R @ ml["E_lambda"]  # (N, K)
        self._value = np.clip(A, 1e-12, np.inf)
