from __future__ import annotations

import numpy as np

from mofapy2.build_model.build_model import buildBiofam

from mofa_subtypes.core.nodes.subtypes_nodes import (
    Z_DP_Node,
    Responsibilities_Node,
    DPGamma_Node,
    DPSticks_Node,
    MuLambda_Node,
    BaseMean_Node,
    T0_Node,
    MuZ_Node,
    AlphaZ_Node,
)


class build_mofa_subtypes(buildBiofam):
    """Builder that augments MOFA with a truncated-DP mixture prior on Z."""

    def build_Z(self):
        # Build standard Z first (PCA initialisation etc.)
        super().build_Z()

        # Replace Z node with Z_DP_Node to get the correct DP-prior ELBO term.
        z = self.init_model.nodes["Z"]
        z_dp = Z_DP_Node(
            dim=z.dim,
            pmean=z.P.getParameters()["mean"],
            pvar=z.P.getParameters()["var"],
            qmean=z.Q.getParameters()["mean"],
            qvar=z.Q.getParameters()["var"],
            qE=z.Q.getExpectations()["E"],
            qE2=z.Q.getExpectations()["E2"],
            weight_views=getattr(z, "weight_views", False),
        )
        self.init_model.nodes["Z"] = z_dp

    def build_nodes(self):
        # Build standard MOFA nodes
        super().build_nodes()

        # Add DP/subtype nodes
        self.build_subtypes_nodes()

    def build_subtypes_nodes(self):
        nodes = self.get_nodes()
        N = self.dim["N"]
        K = self.dim["K"]

        st = self.model_opts.get("subtypes_opts", {})
        C = int(st.get("C", 8))
        assert C >= 2, "C must be >= 2"

        # Hyperparameters (shape-rate for Gammas)
        a_gamma = float(st.get("a_gamma", 1.0))
        b_gamma = float(st.get("b_gamma", 1.0))

        a0 = float(st.get("a0", 1.0))
        b0 = float(st.get("b0", 1.0))

        a_t0 = float(st.get("a_t0", 1.0))
        b_t0 = float(st.get("b_t0", 1.0))

        m0_prior_mean = np.asarray(st.get("m0_prior_mean", np.zeros(K)), dtype=float)
        m0_prior_var = np.asarray(st.get("m0_prior_var", np.ones(K) * 10.0), dtype=float)

        seed = int(self.train_opts.get("seed", 0))

        # Responsibilities
        nodes["C"] = Responsibilities_Node(N=N, C=C, init=st.get("C_init", "uniform"), seed=seed)

        # DP concentration gamma
        nodes["Gamma"] = DPGamma_Node(dim=(1,), pa=a_gamma, pb=b_gamma, qa=a_gamma + (C - 1), qb=b_gamma + 1.0)

        # Sticks v_1..v_{C-1}
        nodes["V"] = DPSticks_Node(
            dim=(C - 1,),
            pa=np.ones(C - 1),
            pb=np.ones(C - 1),
            qa=np.ones(C - 1),
            qb=np.ones(C - 1) * (a_gamma / b_gamma),
        )

        # Component params
        nodes["MuLambda"] = MuLambda_Node(C=C, K=K, a0=a0, b0=b0, seed=seed)

        # Base mean m0 (vector length K)
        nodes["m0"] = BaseMean_Node(
            dim=(K,),
            pmean=m0_prior_mean,
            pvar=m0_prior_var,
            qmean=m0_prior_mean.copy(),
            qvar=m0_prior_var.copy(),
        )

        # t0 (scalar)
        nodes["t0"] = T0_Node(dim=(1,), pa=a_t0, pb=b_t0, qa=a_t0 + 1.0, qb=b_t0 + 1.0)

        # Deterministic per-(n,k) Gaussian prior moments for Z update
        nodes["MuZ"] = MuZ_Node(N=N, K=K)
        nodes["AlphaZ"] = AlphaZ_Node(N=N, K=K)

        # Store for later convenience
        self.model_opts["subtypes_opts"] = {
            **st,
            "C": C,
            "a_gamma": a_gamma,
            "b_gamma": b_gamma,
            "a0": a0,
            "b0": b0,
            "a_t0": a_t0,
            "b_t0": b_t0,
        }

    def createMarkovBlankets(self):
        # Standard MOFA blankets first
        super().createMarkovBlankets()

        nodes = self.get_nodes()

        # DP blankets
        nodes["C"].addMarkovBlanket(V=nodes["V"], Z=nodes["Z"], MuLambda=nodes["MuLambda"])
        nodes["Gamma"].addMarkovBlanket(V=nodes["V"])
        nodes["V"].addMarkovBlanket(C=nodes["C"], Gamma=nodes["Gamma"])
        nodes["MuLambda"].addMarkovBlanket(C=nodes["C"], Z=nodes["Z"], m0=nodes["m0"], t0=nodes["t0"])
        nodes["m0"].addMarkovBlanket(MuLambda=nodes["MuLambda"], t0=nodes["t0"])
        nodes["t0"].addMarkovBlanket(MuLambda=nodes["MuLambda"], m0=nodes["m0"])
        nodes["MuZ"].addMarkovBlanket(C=nodes["C"], MuLambda=nodes["MuLambda"])
        nodes["AlphaZ"].addMarkovBlanket(C=nodes["C"], MuLambda=nodes["MuLambda"])

        # Attach DP-derived prior moments to Z updates and DP context for ELBO
        nodes["Z"].addMarkovBlanket(MuZ=nodes["MuZ"], AlphaZ=nodes["AlphaZ"], C=nodes["C"], MuLambda=nodes["MuLambda"])
