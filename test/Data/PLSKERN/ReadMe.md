# PLSKERN reference fixtures (`generate_plskern_reference.jl`)

The `generate_plskern_reference.jl` is used in context of testing the outputs of `plskern` function of `BigRiverEssence.jl` with the outputs of the Julia implementation of `Jchemo.plskern`. It produces all the simulated data matrices and outputs of `Jchemo.plskern` which are used in `plskern_test.jl` to test similarity of outputs with `Jchemo.plskern`.

## It performs the following tasks:

- It loads the Julia package `Jchemo` and fixes a random seed for reproducibility.
- It simulates a predictor matrix `X`, a single response vector `y` and a multi-response matrix `Y`, and saves them as `X.csv`, `y.csv` and `Y_multi.csv`.
- It fits `Jchemo.plskern` to the `X` matrix with a fixed number of latent variables `nlv`, for both the single response `y` and the multi-response `Y`.
- It writes the outputs of `Jchemo.plskern` for the single response: the regression coefficients `B`, the predictions and the latent scores as `B.csv`, `pred.csv` and `transf.csv` respectively, and for the multi-response: the regression coefficients and predictions as `B_multi.csv` and `pred_multi.csv` respectively.
- It records the `Julia` version and the `Jchemo` package version as `session_info.txt`.
- It creates `meta.csv` containing parameters the fixture was generated it.