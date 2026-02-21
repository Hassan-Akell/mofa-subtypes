import numpy as np
from mofa_subtypes.entry_point import entry_point_subtypes

def main():
    # Toy single-view Gaussian data: Y = ZW^T + noise
    rng = np.random.default_rng(0)
    N, D, K = 200, 50, 5
    Ztrue = rng.normal(size=(N, K))
    Wtrue = rng.normal(size=(D, K))
    Y = Ztrue @ Wtrue.T + 0.5 * rng.normal(size=(N, D))

    data = [Y]  # one view

    ep = entry_point_subtypes()
    ep.set_data_matrix(data)
    ep.set_model_options(factors=K, likelihoods=["gaussian"])
    ep.set_subtypes_options(C=4)
    ep.set_train_options(iter=100, convergence_mode="slow", verbose=True, seed=0)

    ep.build()
    ep.run()
    ep.save("toy_mofa_subtypes.hdf5")

if __name__ == "__main__":
    main()
