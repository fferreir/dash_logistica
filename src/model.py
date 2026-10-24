"""
=============================================================================
Módulo Matemático e Numérico: Modelo de Crescimento Logístico (Gause, 1932)
=============================================================================
Referências:
  Verhulst, P. F. (1838). Notice sur la loi que la population poursuit dans son accroissement.
  Correspondance Mathématique et Physique, 10, 113-121.

  Gause, G. F. (1932). Experimental Studies on the Struggle for Existence:
  1. Mixed Population of Two Species of Yeast.
  Journal of Experimental Biology, 9(4), 389–402.
"""

import numpy as np
from scipy.integrate import solve_ivp

# =============================================================================
# DADOS EXPERIMENTAIS CLÁSSICOS DE G. F. GAUSE (1932)
# Tempo em horas (h); biomassa/população em unidades de volume de levedura.
# =============================================================================
T_SAC_REAL = np.array([6.0, 7.5, 15.0, 16.0, 24.0, 24.0, 29.0, 31.5, 33.0, 40.0, 44.0, 48.0, 51.5, 53.0])
SAC_REAL = np.array([0.37, 1.63, 6.20, 8.87, 10.66, 10.97, 12.50, 12.60, 12.90, 13.27, 12.77, 12.87, 12.90, 12.70])

T_SCH_REAL = np.array([15.0, 16.0, 29.0, 31.5, 48.0, 51.5, 72.0, 93.0, 117.0, 141.0])
SCH_REAL = np.array([1.27, 1.00, 1.70, 2.33, 2.73, 4.56, 4.87, 5.67, 5.80, 5.83])


def ode_sys(t, state, r_sac, k_sac, r_sch, k_sch):
    """
    Sistema desacoplado de equações diferenciais ordinárias para crescimento logístico.
    dN/dt = r * N * (1 - N / K)
    
    Parâmetros:
      t: Tempo em horas
      state: [sac, sch] (volumes relativos de levedura)
      r_sac: Taxa intrínseca de Saccharomyces (h⁻¹)
      k_sac: Capacidade de suporte de Saccharomyces (vol)
      r_sch: Taxa intrínseca de Schizosaccharomyces (h⁻¹)
      k_sch: Capacidade de suporte de Schizosaccharomyces (vol)
      
    Retorna:
      [dsac_dt, dsch_dt]
    """
    sac = max(0.0, float(state[0]))
    sch = max(0.0, float(state[1]))
    
    k_sac_eff = max(float(k_sac), 1e-4)
    k_sch_eff = max(float(k_sch), 1e-4)
    
    dsac_dt = float(r_sac) * sac * (1.0 - (sac / k_sac_eff))
    dsch_dt = float(r_sch) * sch * (1.0 - (sch / k_sch_eff))
    
    return [dsac_dt, dsch_dt]


def solucao_analitica(t_arr, r, k, n0):
    """
    Solução analítica exata da equação logística de Verhulst:
    N(t) = K / (1 + ((K - N0) / N0) * exp(-r * t))
    """
    k_eff = max(float(k), 1e-4)
    n0_eff = max(float(n0), 1e-4)
    r_eff = float(r)
    
    termo_exp = ((k_eff - n0_eff) / n0_eff) * np.exp(-r_eff * np.asarray(t_arr, dtype=float))
    return k_eff / (1.0 + termo_exp)


def calcular_metricas_cineticas(r_sac, k_sac, n0_sac, r_sch, k_sch, n0_sch):
    """
    Calcula as grandezas cinéticas, tempos característicos e indicadores de ajuste (R² e RMSE)
    frente aos dados experimentais originais de Gause (1932).
    """
    r_sac_val = max(float(r_sac), 1e-5)
    k_sac_val = max(float(k_sac), 1e-4)
    n0_sac_val = max(float(n0_sac), 1e-4)

    r_sch_val = max(float(r_sch), 1e-5)
    k_sch_val = max(float(k_sch), 1e-4)
    n0_sch_val = max(float(n0_sch), 1e-4)

    # Tempos de duplicação na fase exponencial: td = ln(2) / r
    td_sac = np.log(2.0) / r_sac_val
    td_sch = np.log(2.0) / r_sch_val

    # Tempo de inflexão (N = K/2, onde d²N/dt² = 0)
    if k_sac_val > n0_sac_val:
        t_inflex_sac = (1.0 / r_sac_val) * np.log((k_sac_val - n0_sac_val) / n0_sac_val)
    else:
        t_inflex_sac = 0.0

    if k_sch_val > n0_sch_val:
        t_inflex_sch = (1.0 / r_sch_val) * np.log((k_sch_val - n0_sch_val) / n0_sch_val)
    else:
        t_inflex_sch = 0.0

    # Taxa máxima de crescimento instantâneo: (dN/dt)_max = r * K / 4
    max_taxa_sac = (r_sac_val * k_sac_val) / 4.0
    max_taxa_sch = (r_sch_val * k_sch_val) / 4.0

    # Avaliação do ajuste (Goodness of Fit) contra dados reais de Gause
    pred_sac_real = solucao_analitica(T_SAC_REAL, r_sac_val, k_sac_val, n0_sac_val)
    rmse_sac = float(np.sqrt(np.mean((SAC_REAL - pred_sac_real) ** 2)))
    ss_tot_sac = float(np.sum((SAC_REAL - np.mean(SAC_REAL)) ** 2))
    ss_res_sac = float(np.sum((SAC_REAL - pred_sac_real) ** 2))
    r2_sac = max(0.0, 1.0 - (ss_res_sac / ss_tot_sac)) if ss_tot_sac > 1e-9 else 0.0

    pred_sch_real = solucao_analitica(T_SCH_REAL, r_sch_val, k_sch_val, n0_sch_val)
    rmse_sch = float(np.sqrt(np.mean((SCH_REAL - pred_sch_real) ** 2)))
    ss_tot_sch = float(np.sum((SCH_REAL - np.mean(SCH_REAL)) ** 2))
    ss_res_sch = float(np.sum((SCH_REAL - pred_sch_real) ** 2))
    r2_sch = max(0.0, 1.0 - (ss_res_sch / ss_tot_sch)) if ss_tot_sch > 1e-9 else 0.0

    return {
        "td_sac": td_sac,
        "td_sch": td_sch,
        "t_inflex_sac": t_inflex_sac,
        "t_inflex_sch": t_inflex_sch,
        "max_taxa_sac": max_taxa_sac,
        "max_taxa_sch": max_taxa_sch,
        "r2_sac": r2_sac,
        "r2_sch": r2_sch,
        "rmse_sac": rmse_sac,
        "rmse_sch": rmse_sch,
    }


def simular_logistica(r_sac, k_sac, n0_sac, r_sch, k_sch, n0_sch, t_max=150.0, n_points=1200):
    """
    Executa a simulação temporal analítica e o cálculo das derivadas dN/dt
    para as monoculturas de Saccharomyces cerevisiae e Schizosaccharomyces kephir.
    
    Retorna:
      dicionário com séries temporais de população, derivadas instantâneas e métricas cinéticas.
    """
    r_sac_val = float(r_sac) if r_sac is not None else 0.23
    k_sac_val = float(k_sac) if k_sac is not None else 13.0
    n0_sac_val = float(n0_sac) if n0_sac is not None else 0.45

    r_sch_val = float(r_sch) if r_sch is not None else 0.060
    k_sch_val = float(k_sch) if k_sch is not None else 5.8
    n0_sch_val = float(n0_sch) if n0_sch is not None else 0.46

    t_end = max(1.0, float(t_max) if t_max is not None else 150.0)
    t_eval = np.linspace(0.0, t_end, int(n_points))

    # Séries temporais de população
    sac_eval = solucao_analitica(t_eval, r_sac_val, k_sac_val, n0_sac_val)
    sch_eval = solucao_analitica(t_eval, r_sch_val, k_sch_val, n0_sch_val)

    # Derivadas instantâneas de crescimento (dN/dt)
    k_sac_eff = max(k_sac_val, 1e-4)
    k_sch_eff = max(k_sch_val, 1e-4)
    dsac_dt_eval = r_sac_val * sac_eval * (1.0 - (sac_eval / k_sac_eff))
    dsch_dt_eval = r_sch_val * sch_eval * (1.0 - (sch_eval / k_sch_eff))

    metricas = calcular_metricas_cineticas(
        r_sac_val, k_sac_val, n0_sac_val,
        r_sch_val, k_sch_val, n0_sch_val
    )

    return {
        "t": t_eval,
        "sac": sac_eval,
        "sch": sch_eval,
        "dsac_dt": dsac_dt_eval,
        "dsch_dt": dsch_dt_eval,
        "r_sac": r_sac_val,
        "k_sac": k_sac_val,
        "n0_sac": n0_sac_val,
        "r_sch": r_sch_val,
        "k_sch": k_sch_val,
        "n0_sch": n0_sch_val,
        "t_end": t_end,
        **metricas
    }
