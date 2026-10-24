"""
Testes unitários para o modelo matemático e numérico logístico (Gause, 1932).
"""

import pytest
import numpy as np
from src.model import (
    ode_sys,
    solucao_analitica,
    calcular_metricas_cineticas,
    simular_logistica,
    T_SAC_REAL,
    SAC_REAL,
    T_SCH_REAL,
    SCH_REAL,
)


def test_ode_sys_extinction_zeros():
    """Se as populações forem zero, as taxas de crescimento devem ser nulas."""
    state = [0.0, 0.0]
    derivs = ode_sys(0.0, state, r_sac=0.23, k_sac=13.0, r_sch=0.06, k_sch=5.8)
    assert derivs == [0.0, 0.0]


def test_ode_sys_saturation_k():
    """Quando as populações atingem K, as derivadas dN/dt devem ser nulas."""
    state = [13.0, 5.8]
    derivs = ode_sys(0.0, state, r_sac=0.23, k_sac=13.0, r_sch=0.06, k_sch=5.8)
    assert pytest.approx(derivs[0], 1e-6) == 0.0
    assert pytest.approx(derivs[1], 1e-6) == 0.0


def test_solucao_analitica_initial_condition():
    """Em t = 0, a solução analítica deve ser exatamente igual a N0."""
    t0 = np.array([0.0])
    n_sac_0 = solucao_analitica(t0, r=0.23, k=13.0, n0=0.45)
    n_sch_0 = solucao_analitica(t0, r=0.06, k=5.8, n0=0.46)
    
    assert pytest.approx(n_sac_0[0], 1e-6) == 0.45
    assert pytest.approx(n_sch_0[0], 1e-6) == 0.46


def test_solucao_analitica_asymptote():
    """Para tempos muito longos (t -> inf), a população deve convergir para K."""
    t_long = np.array([500.0])
    n_sac_inf = solucao_analitica(t_long, r=0.23, k=13.0, n0=0.45)
    n_sch_inf = solucao_analitica(t_long, r=0.06, k=5.8, n0=0.46)
    
    assert pytest.approx(n_sac_inf[0], 1e-3) == 13.0
    assert pytest.approx(n_sch_inf[0], 1e-3) == 5.8


def test_metricas_cineticas_gause_standards():
    """Valida as grandezas cinéticas e o coeficiente de determinação R² com os dados de Gause."""
    m = calcular_metricas_cineticas(
        r_sac=0.23, k_sac=13.0, n0_sac=0.45,
        r_sch=0.060, k_sch=5.8, n0_sch=0.46
    )
    
    # Tempo de duplicação: td = ln(2) / r
    assert pytest.approx(m["td_sac"], 0.1) == 3.01
    assert pytest.approx(m["td_sch"], 0.1) == 11.55
    
    # Instante de inflexão: t* = (1/r) * ln((K - N0) / N0)
    # Sac: (1/0.23) * ln((13 - 0.45)/0.45) = (1/0.23) * ln(27.888) ≈ 14.47 h
    assert pytest.approx(m["t_inflex_sac"], 0.2) == 14.5
    # Sch: (1/0.06) * ln((5.8 - 0.46)/0.46) = (1/0.06) * ln(11.608) ≈ 40.86 h
    assert pytest.approx(m["t_inflex_sch"], 0.2) == 40.9
    
    # Ajuste aos dados de Gause: R² deve ser alto (> 0.95)
    assert m["r2_sac"] > 0.95
    assert m["r2_sch"] > 0.95
    assert m["rmse_sac"] < 1.0
    assert m["rmse_sch"] < 0.6


def test_simulacao_monotonicidade_e_estrutura():
    """Simulação padrão com N0 < K deve ser monotonicamente crescente e retornar todas as chaves."""
    res = simular_logistica(
        r_sac=0.23, k_sac=13.0, n0_sac=0.45,
        r_sch=0.06, k_sch=5.8, n0_sch=0.46,
        t_max=150.0, n_points=500
    )
    
    assert len(res["t"]) == 500
    assert len(res["sac"]) == 500
    assert len(res["sch"]) == 500
    
    # Monotonicidade estrita para N0 < K
    assert np.all(np.diff(res["sac"]) >= 0.0)
    assert np.all(np.diff(res["sch"]) >= 0.0)
    
    # Valores finais próximos de K
    assert pytest.approx(res["sac"][-1], 0.05) == 13.0
    assert pytest.approx(res["sch"][-1], 0.05) == 5.8


def test_supersaturacao_declinio():
    """Quando N0 > K, a população deve declinar em direção a K (dN/dt < 0)."""
    res = simular_logistica(
        r_sac=0.23, k_sac=13.0, n0_sac=20.0,
        r_sch=0.06, k_sch=5.8, n0_sch=10.0,
        t_max=50.0, n_points=200
    )
    
    # Curva deve ser estritamente decrescente
    assert np.all(np.diff(res["sac"]) <= 0.0)
    assert np.all(np.diff(res["sch"]) <= 0.0)
    assert res["sac"][-1] < 20.0
    assert pytest.approx(res["sac"][-1], 0.1) == 13.0
