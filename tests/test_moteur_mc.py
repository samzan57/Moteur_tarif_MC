"""
Unit tests for the Monte Carlo Asian option pricer (moteur_mc.py).

The headline test validates the engine against a real analytical
benchmark: for a GEOMETRIC-average Asian option, a closed-form price
exists (the geometric average of a GBM is itself log-normal), unlike
the arithmetic-average case this project targets. Matching that
closed form is strong evidence the simulation, discounting and payoff
logic are all correct.

Run with:
    pytest tests/
"""

import os
import sys

import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from moteur_mc import pricer_option_asiatique, simuler_trajectoires, payoff_asiatique


def _closed_form_geometric_asian_call(S0, K, r, sigma, T, n_monitoring):
    """
    Closed-form price of a fixed-strike geometric-average Asian call,
    discretely monitored at n_monitoring equally spaced dates.

    ln(G) = mean of ln(S(t_i)) is Gaussian; the call price follows a
    Black-Scholes-style formula on that Gaussian (Kemna-Vorst, 1990,
    with the discrete-monitoring variance correction).
    """
    n = n_monitoring
    mean_log = np.log(S0) + (r - 0.5 * sigma**2) * T * (n + 1) / (2 * n)
    var_log = sigma**2 * T * (n + 1) * (2 * n + 1) / (6 * n**2)
    vol = np.sqrt(var_log)

    d1 = (mean_log - np.log(K) + var_log) / vol
    d2 = d1 - vol
    return np.exp(-r * T) * (np.exp(mean_log + 0.5 * var_log) * norm.cdf(d1) - K * norm.cdf(d2))


def test_geometric_asian_call_matches_closed_form():
    S0, K, r, sigma, T, N_pas = 100.0, 100.0, 0.05, 0.20, 1.0, 50

    result = pricer_option_asiatique(
        S0, K, r, sigma, T, N_pas=N_pas, N_simul=20_000,
        type_option="call", type_moyenne="geometrique", seed=7,
    )
    closed_form = _closed_form_geometric_asian_call(S0, K, r, sigma, T, N_pas)

    # Ecart attendu de l'ordre de l'erreur standard Monte Carlo -> marge large
    # pour eviter un test qui echoue par pur hasard, tout en restant strict :
    # une vraie erreur de simulation/actualisation donnerait un ecart bien plus grand.
    assert abs(result["prix"] - closed_form) < 5 * result["erreur_std"]


def test_confidence_interval_contains_price():
    res = pricer_option_asiatique(100.0, 100.0, 0.05, 0.20, 1.0, N_simul=5_000, seed=1)
    assert res["ic_bas"] <= res["prix"] <= res["ic_haut"]


def test_call_price_increases_with_volatility():
    low = pricer_option_asiatique(100.0, 100.0, 0.05, 0.10, 1.0, N_simul=20_000, seed=2)
    high = pricer_option_asiatique(100.0, 100.0, 0.05, 0.40, 1.0, N_simul=20_000, seed=2)
    assert high["prix"] > low["prix"]


def test_deep_out_of_the_money_call_is_near_zero():
    res = pricer_option_asiatique(100.0, 500.0, 0.05, 0.20, 1.0, N_simul=5_000, seed=3)
    assert res["prix"] < 0.01


def test_standard_error_shrinks_as_simulations_increase():
    small = pricer_option_asiatique(100.0, 100.0, 0.05, 0.20, 1.0, N_simul=1_000, seed=4)
    large = pricer_option_asiatique(100.0, 100.0, 0.05, 0.20, 1.0, N_simul=20_000, seed=4)
    assert large["erreur_std"] < small["erreur_std"]


def test_simuler_trajectoires_shape_and_initial_price():
    trajs = simuler_trajectoires(S0=100.0, mu=0.05, sigma=0.2, T=1.0, N_pas=50, N_simul=200, seed=0)
    assert trajs.shape == (200, 51)
    assert np.allclose(trajs[:, 0], 100.0)


def test_payoff_asiatique_call_vs_put():
    trajs = simuler_trajectoires(S0=100.0, mu=0.05, sigma=0.2, T=1.0, N_pas=50, N_simul=500, seed=0)
    call_payoffs = payoff_asiatique(trajs, K=100.0, type_option="call")
    put_payoffs = payoff_asiatique(trajs, K=100.0, type_option="put")
    assert (call_payoffs >= 0).all()
    assert (put_payoffs >= 0).all()
