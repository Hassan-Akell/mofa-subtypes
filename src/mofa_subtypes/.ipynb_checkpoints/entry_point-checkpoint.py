from __future__ import annotations

import os
import h5py
import numpy as np

from mofapy2.run.entry_point import entry_point as mofa_entry_point
from mofapy2.core.BayesNet import BayesNet, StochasticBayesNet

from mofa_subtypes.build_model import build_mofa_subtypes


class entry_point_subtypes(mofa_entry_point):
    """MOFA entry point extended with DP-Cluster-MOFA (subtype) components."""

    def set_subtypes_options(
        self,
        C: int = 8,
        a_gamma: float = 1.0,
        b_gamma: float = 1.0,
        a0: float = 1.0,
        b0: float = 1.0,
        a_t0: float = 1.0,
        b_t0: float = 1.0,
        m0_prior_mean=None,
        m0_prior_var=None,
        C_init: str = "uniform",
    ):
        """
        Configure the truncated-DP mixture prior on Z.

        Parameters
        ----------
        C : int
            Truncation level (number of mixture components).
        a_gamma, b_gamma : float
            Gamma prior on DP concentration (shape-rate).
        a0, b0 : float
            Gamma base prior for component precisions lambda_{ck} (shape-rate).
        a_t0, b_t0 : float
            Gamma prior for t0 (shape-rate).
        m0_prior_mean, m0_prior_var : array-like of length K
            Prior mean/variance for m0k. If None, set to 0 and 10.
        C_init : str
            "uniform" or "random".
        """
        if not hasattr(self, "model_opts"):
            raise ValueError("Call set_model_options(...) before set_subtypes_options(...)")
        K = self.model_opts["factors"]

        if m0_prior_mean is None:
            m0_prior_mean = np.zeros(K)
        if m0_prior_var is None:
            m0_prior_var = np.ones(K) * 10.0

        self.model_opts["subtypes_opts"] = {
            "C": int(C),
            "a_gamma": float(a_gamma),
            "b_gamma": float(b_gamma),
            "a0": float(a0),
            "b0": float(b0),
            "a_t0": float(a_t0),
            "b_t0": float(b_t0),
            "m0_prior_mean": np.asarray(m0_prior_mean, dtype=float),
            "m0_prior_var": np.asarray(m0_prior_var, dtype=float),
            "C_init": C_init,
        }

    # def set_train_options(self, *args, **kwargs):
    #     # Let mofapy2 parse standard options first
    #     super().set_train_options(*args, **kwargs)

    #     # If subtypes are enabled, provide a sensible default schedule
    #     if hasattr(self, "model_opts") and "subtypes_opts" in self.model_opts:
    #         if self.train_opts.get("schedule", None) is None:
    #             self.train_opts["schedule"] = [
    #                 "Y",
    #                 "W",
    #                 "Z",
    #                 "Tau",
    #                 # Subtypes block
    #                 "C",
    #                 "Gamma",
    #                 "V",
    #                 "MuLambda",
    #                 "m0",
    #                 "t0",
    #                 "MuZ",
    #                 "AlphaZ",
    #             ]

                #I need to decide whether to select the option above or below of set_train_options.
                
    def set_train_options(self, *args, **kwargs):
        super().set_train_options(*args, **kwargs)

        if hasattr(self, "model_opts") and "subtypes_opts" in self.model_opts:
            extra = ["C","Gamma","V","MuLambda","m0","t0","MuZ","AlphaZ"]
            sch = list(self.train_opts.get("schedule", []))
            for x in extra:
                if x not in sch:
                    sch.append(x)
            self.train_opts["schedule"] = sch

    
    def build(self):
        """Build the model (MOFA + DP subtypes)."""

        # Sanity checks from base class
        assert hasattr(self, "train_opts"), "Training options not defined"
        assert hasattr(self, "model_opts"), "Model options not defined"
        assert hasattr(self, "dimensionalities"), "Dimensionalities are not defined"

        # Build nodes
        tmp = build_mofa_subtypes(
            self.data,
            self.dimensionalities,
            self.data_opts,
            self.model_opts,
            self.train_opts,
        )
        tmp.main()
        
        self.model_builder = tmp

        # Create BayesNet
        if self.train_opts["stochastic"]:
            self.model = StochasticBayesNet(self.dimensionalities, tmp.get_nodes())
        else:
            self.model = BayesNet(self.dimensionalities, tmp.get_nodes())

    def save(self, outfile=None, save_data=True, save_parameters=False, expectations=None):
        # Use base save (writes standard MOFA artefacts)
        super().save(outfile=outfile, save_data=save_data, save_parameters=save_parameters, expectations=expectations)

        # Append subtype artefacts if present
        outfile = outfile or self.train_opts.get("outfile")
        if not outfile or not os.path.isfile(outfile):
            return

        nodes = getattr(self.model, "nodes", {})
        if not all(k in nodes for k in ["C", "V", "Gamma", "MuLambda", "m0", "t0"]):
            return

        with h5py.File(outfile, "a") as f:
            grp = f.require_group("subtypes")
            # Overwrite safely
            for k in list(grp.keys()):
                del grp[k]

            R = nodes["C"].Q.getExpectations()["E"]
            grp.create_dataset("responsibilities", data=R)

            Vexp = nodes["V"].Q.getExpectations()
            grp.create_dataset("sticks_E", data=Vexp["E"])
            grp.create_dataset("sticks_lnE", data=Vexp["lnE"])
            grp.create_dataset("sticks_lnEInv", data=Vexp["lnEInv"])

            grp.create_dataset("gamma_E", data=nodes["Gamma"].Q.getExpectations()["E"])

            ml = nodes["MuLambda"].Q.getExpectations()
            grp.create_dataset("mu", data=ml["E_mu"])
            grp.create_dataset("lambda_E", data=ml["E_lambda"])
            grp.create_dataset("lambda_logE", data=ml["E_log_lambda"])

            grp.create_dataset("m0", data=nodes["m0"].Q.getParameters()["mean"])
            grp.create_dataset("t0_E", data=nodes["t0"].Q.getExpectations()["E"])

            # Store options
            opts = self.model_opts.get("subtypes_opts", {})
            for key, val in opts.items():
                if np.isscalar(val):
                    grp.attrs[key] = val
