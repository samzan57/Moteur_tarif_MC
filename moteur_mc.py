"""
Moteur de tarification Monte Carlo — Option Asiatique
======================================================
Une option asiatique dépend de la MOYENNE du cours de l'action sur toute la durée de vie
de l'option, pas seulement du cours final. Pas de formule fermée simple → Monte Carlo.

Modèle sous-jacent : Mouvement Brownien Géométrique (GBM)
    dS = mu * S * dt + sigma * S * dW
Forme discrète (schéma d'Euler-Maruyama log-normal) :
    S(t+dt) = S(t) * exp((mu - 0.5*sigma²)*dt + sigma*sqrt(dt)*Z)   Z ~ N(0,1)
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. SIMULATION DES TRAJECTOIRES
# ---------------------------------------------------------------------------

def simuler_trajectoires(S0, mu, sigma, T, N_pas, N_simul, seed=None):
    """
    Simule N_simul trajectoires du cours d'une action selon le GBM.

    Paramètres
    ----------
    S0      : float  — cours initial de l'action
    mu      : float  — taux de dérive (ex: taux sans risque r pour pricing risque-neutre)
    sigma   : float  — volatilité annuelle
    T       : float  — maturité en années
    N_pas   : int    — nombre de pas de temps par trajectoire
    N_simul : int    — nombre de simulations (trajectoires)
    seed    : int    — graine aléatoire pour reproductibilité

    Retourne
    --------
    trajectoires : np.ndarray de forme (N_simul, N_pas+1)
                   trajectoires[i, j] = cours de l'action à l'instant j pour la sim i
    """
    rng = np.random.default_rng(seed)

    dt = T / N_pas

    # Tirage de tous les chocs aléatoires en une seule fois (vectorisé)
    Z = rng.standard_normal((N_simul, N_pas))  # (N_simul × N_pas) gaussiennes

    # Incrément log-normal : log(S(t+dt)/S(t)) = (mu - σ²/2)*dt + σ*√dt*Z
    increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    # Trajectoires par cumul des log-rendements
    log_trajectoires = np.cumsum(increments, axis=1)               # (N_simul, N_pas)
    log_trajectoires = np.hstack([np.zeros((N_simul, 1)),          # ajouter S0 au début
                                   log_trajectoires])

    trajectoires = S0 * np.exp(log_trajectoires)                   # (N_simul, N_pas+1)
    return trajectoires


# ---------------------------------------------------------------------------
# 2. CALCUL DU PAYOFF ASIATIQUE
# ---------------------------------------------------------------------------

def payoff_asiatique(trajectoires, K, type_option="call", type_moyenne="arithmetique"):
    """
    Calcule le payoff d'une option asiatique à prix d'exercice fixe (fixed-strike).

    Payoff call : max(S_moy - K, 0)
    Payoff put  : max(K - S_moy, 0)

    Paramètres
    ----------
    trajectoires   : np.ndarray (N_simul, N_pas+1) — trajectoires simulées
    K              : float — prix d'exercice (strike)
    type_option    : "call" ou "put"
    type_moyenne   : "arithmetique" ou "geometrique"

    Retourne
    --------
    payoffs : np.ndarray (N_simul,) — payoff pour chaque simulation
    """
    # Calcul de la moyenne du cours sur toute la trajectoire (en excluant S0)
    if type_moyenne == "arithmetique":
        S_moy = np.mean(trajectoires[:, 1:], axis=1)   # moyenne simple
    elif type_moyenne == "geometrique":
        log_moy = np.mean(np.log(trajectoires[:, 1:]), axis=1)
        S_moy = np.exp(log_moy)                         # exp(moyenne des log)
    else:
        raise ValueError(f"type_moyenne inconnu : {type_moyenne}")

    if type_option == "call":
        payoffs = np.maximum(S_moy - K, 0.0)
    elif type_option == "put":
        payoffs = np.maximum(K - S_moy, 0.0)
    else:
        raise ValueError(f"type_option inconnu : {type_option}")

    return payoffs


# ---------------------------------------------------------------------------
# 3. MOTEUR DE PRICING PRINCIPAL
# ---------------------------------------------------------------------------

def pricer_option_asiatique(S0, K, r, sigma, T,
                            N_pas=252, N_simul=100_000,
                            type_option="call", type_moyenne="arithmetique",
                            seed=42):
    """
    Prix d'une option asiatique par Monte Carlo.

    Le prix est l'espérance actualisée des payoffs sous la mesure risque-neutre :
        Prix = e^(-r*T) * E[Payoff]

    On estime aussi l'intervalle de confiance à 95 % pour quantifier l'erreur MC.

    Paramètres
    ----------
    S0          : float — cours initial
    K           : float — strike
    r           : float — taux sans risque (annuel, continu)
    sigma       : float — volatilité annuelle
    T           : float — maturité (années)
    N_pas       : int   — pas de discrétisation (252 = jours ouvrés/an)
    N_simul     : int   — nombre de simulations
    type_option : "call" ou "put"
    type_moyenne: "arithmetique" ou "geometrique"
    seed        : int   — reproductibilité

    Retourne
    --------
    dict avec : prix, erreur_std, ic_bas, ic_haut, trajectoires
    """
    # --- Simulation (risque-neutre : mu = r) ---
    trajectoires = simuler_trajectoires(S0, mu=r, sigma=sigma, T=T,
                                        N_pas=N_pas, N_simul=N_simul, seed=seed)

    # --- Payoffs ---
    payoffs = payoff_asiatique(trajectoires, K,
                               type_option=type_option,
                               type_moyenne=type_moyenne)

    # --- Actualisation ---
    facteur_actu = np.exp(-r * T)
    payoffs_actualises = facteur_actu * payoffs

    # --- Statistiques Monte Carlo ---
    prix = np.mean(payoffs_actualises)
    erreur_std = np.std(payoffs_actualises) / np.sqrt(N_simul)  # erreur standard
    ic_bas  = prix - 1.96 * erreur_std   # borne inf IC 95%
    ic_haut = prix + 1.96 * erreur_std   # borne sup IC 95%

    return {
        "prix":         prix,
        "erreur_std":   erreur_std,
        "ic_bas":       ic_bas,
        "ic_haut":      ic_haut,
        "trajectoires": trajectoires,
        "payoffs":      payoffs,
    }


# ---------------------------------------------------------------------------
# 4. ANALYSE DE CONVERGENCE
# ---------------------------------------------------------------------------

def analyser_convergence(S0, K, r, sigma, T,
                         liste_N=[1_000, 5_000, 10_000, 50_000, 100_000],
                         type_option="call", seed=42):
    """
    Montre comment le prix MC converge vers la vraie valeur quand N_simul augmente.
    Retourne une liste de (N_simul, prix, erreur_std).
    """
    resultats = []
    for N in liste_N:
        res = pricer_option_asiatique(S0, K, r, sigma, T,
                                      N_simul=N, type_option=type_option, seed=seed)
        resultats.append({
            "N_simul":    N,
            "prix":       res["prix"],
            "erreur_std": res["erreur_std"],
            "ic_bas":     res["ic_bas"],
            "ic_haut":    res["ic_haut"],
        })
        print(f"  N={N:>8,} | Prix = {res['prix']:.4f} | "
              f"IC 95% = [{res['ic_bas']:.4f}, {res['ic_haut']:.4f}]")
    return resultats


# ---------------------------------------------------------------------------
# 5. VISUALISATION
# ---------------------------------------------------------------------------

def visualiser(res, S0, K, T, r, sigma, n_trajs=50):
    """
    3 graphiques :
      1. Trajectoires simulées + moyenne
      2. Distribution des cours moyens (S_moy) avec strike K
      3. Convergence du prix MC selon N_simul
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Moteur Monte Carlo — Option Asiatique", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    trajs = res["trajectoires"]           # (N_simul, N_pas+1)
    N_pas = trajs.shape[1] - 1
    t = np.linspace(0, T, N_pas + 1)

    # --- Graphique 1 : trajectoires simulées ---
    ax1 = fig.add_subplot(gs[0, 0])
    idx = np.random.choice(len(trajs), n_trajs, replace=False)
    for i in idx:
        ax1.plot(t, trajs[i], color="steelblue", alpha=0.25, linewidth=0.7)
    ax1.plot(t, np.mean(trajs, axis=0), color="crimson", linewidth=2, label="Moyenne MC")
    ax1.axhline(K, color="black", linestyle="--", linewidth=1, label=f"Strike K={K}")
    ax1.set_title(f"{n_trajs} trajectoires simulees (GBM)")
    ax1.set_xlabel("Temps (annees)")
    ax1.set_ylabel("Cours de l'action ($)")
    ax1.legend(fontsize=8)

    # --- Graphique 2 : distribution de S_moy ---
    ax2 = fig.add_subplot(gs[0, 1])
    S_moy = np.mean(trajs[:, 1:], axis=1)
    ax2.hist(S_moy, bins=80, color="steelblue", edgecolor="white", linewidth=0.3, density=True)
    ax2.axvline(K, color="crimson", linestyle="--", linewidth=2, label=f"Strike K={K}")
    ax2.axvline(np.mean(S_moy), color="orange", linestyle="-", linewidth=2,
                label=f"Moyenne={np.mean(S_moy):.1f}")
    ax2.set_title("Distribution des cours moyens S_moy")
    ax2.set_xlabel("Cours moyen ($)")
    ax2.set_ylabel("Densite")
    ax2.legend(fontsize=8)

    # --- Graphique 3 : distribution des payoffs ---
    ax3 = fig.add_subplot(gs[1, 0])
    payoffs = res["payoffs"]
    payoffs_pos = payoffs[payoffs > 0]
    ax3.hist(payoffs_pos, bins=60, color="seagreen", edgecolor="white", linewidth=0.3, density=True)
    ax3.axvline(res["prix"] * np.exp(r * T), color="crimson", linestyle="--", linewidth=2,
                label=f"Payoff moyen={np.mean(payoffs_pos):.2f}")
    ax3.set_title(f"Distribution des payoffs > 0  ({100*len(payoffs_pos)/len(payoffs):.1f}% dans la monnaie)")
    ax3.set_xlabel("Payoff ($)")
    ax3.set_ylabel("Densite")
    ax3.legend(fontsize=8)

    # --- Graphique 4 : convergence du prix ---
    ax4 = fig.add_subplot(gs[1, 1])
    liste_N = [500, 1_000, 2_000, 5_000, 10_000, 30_000, 100_000]
    prix_conv, ic_bas_conv, ic_haut_conv = [], [], []
    for N in liste_N:
        r2 = pricer_option_asiatique(S0, K, r, sigma, T, N_simul=N, seed=0)
        prix_conv.append(r2["prix"])
        ic_bas_conv.append(r2["ic_bas"])
        ic_haut_conv.append(r2["ic_haut"])
    ax4.semilogx(liste_N, prix_conv, "o-", color="steelblue", linewidth=2, label="Prix MC")
    ax4.fill_between(liste_N, ic_bas_conv, ic_haut_conv, alpha=0.25, color="steelblue", label="IC 95%")
    ax4.axhline(res["prix"], color="crimson", linestyle="--", linewidth=1.5, label=f"Ref={res['prix']:.3f}")
    ax4.set_title("Convergence du prix selon N simulations")
    ax4.set_xlabel("Nombre de simulations (echelle log)")
    ax4.set_ylabel("Prix ($)")
    ax4.legend(fontsize=8)

    plt.savefig("resultats_mc.png", dpi=150, bbox_inches="tight")
    print("\nGraphiques sauvegardes dans : resultats_mc.png")
    plt.show()


# ---------------------------------------------------------------------------
# 6. DEMO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  MOTEUR DE TARIFICATION MONTE CARLO — OPTION ASIATIQUE")
    print("=" * 60)

    # Paramètres de l'exemple
    S0    = 100.0    # cours initial
    K     = 100.0    # strike (at-the-money)
    r     = 0.05     # taux sans risque 5%
    sigma = 0.20     # volatilité 20%
    T     = 1.0      # maturité 1 an

    print(f"\nParametres : S0={S0}, K={K}, r={r:.0%}, sigma={sigma:.0%}, T={T}an")

    # --- Pricing call asiatique (moyenne arithmétique) ---
    print("\n[ Call Asiatique — Moyenne Arithmétique ]")
    res_call = pricer_option_asiatique(S0, K, r, sigma, T,
                                       N_pas=252, N_simul=200_000,
                                       type_option="call",
                                       type_moyenne="arithmetique")
    print(f"  Prix MC     : {res_call['prix']:.4f} $")
    print(f"  Erreur std  : {res_call['erreur_std']:.4f} $")
    print(f"  IC 95%      : [{res_call['ic_bas']:.4f}, {res_call['ic_haut']:.4f}]")

    # --- Pricing put asiatique ---
    print("\n[ Put Asiatique — Moyenne Arithmétique ]")
    res_put = pricer_option_asiatique(S0, K, r, sigma, T,
                                      N_pas=252, N_simul=200_000,
                                      type_option="put",
                                      type_moyenne="arithmetique")
    print(f"  Prix MC     : {res_put['prix']:.4f} $")
    print(f"  IC 95%      : [{res_put['ic_bas']:.4f}, {res_put['ic_haut']:.4f}]")

    # --- Vérification parité call-put asiatique (approximative) ---
    print("\n[ Vérification parité call - put ]")
    parite = res_call["prix"] - res_put["prix"]
    print(f"  C - P = {parite:.4f}  "
          f"(théoriquement ≈ S0·e^(-r·T)·(S_moy_fwd/K - 1) pour option asiatique)")

    # --- Analyse de convergence ---
    print("\n[ Analyse de convergence ]")
    analyser_convergence(S0, K, r, sigma, T, type_option="call")

    print("\nTerminé.")

    # --- Visualisations ---
    visualiser(res_call, S0, K, T, r, sigma)
