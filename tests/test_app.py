"""
Testes de integração para a interface, rodapé e callbacks do Dash Logística.
"""

import pytest
import numpy as np
from app import app
from src.model import simular_logistica, calcular_metricas_cineticas


def test_app_layout_structure():
    """Verifica se o layout do Dash é criado e não é nulo."""
    layout = app.layout
    assert layout is not None


def test_footer_github_link():
    """Verifica se o rodapé institucional possui o link correto para o repositório GitHub."""
    from src.layout import rodape
    assert rodape is not None
    footer_repr = str(rodape)
    assert "https://github.com/fferreir/dash_logistica" in footer_repr


def test_simulacao_parametros_extremos():
    """Verifica estabilidade numérica sob parâmetros extremos."""
    res = simular_logistica(
        r_sac=0.60, k_sac=25.0, n0_sac=0.05,
        r_sch=0.30, k_sch=15.0, n0_sch=0.05,
        t_max=300.0, n_points=500
    )
    assert not np.isnan(res['sac']).any()
    assert not np.isnan(res['sch']).any()
    assert not np.isinf(res['sac']).any()
    assert not np.isinf(res['sch']).any()
    assert (res['sac'] >= 0.0).all()
    assert (res['sch'] >= 0.0).all()


def test_calcular_metricas_limites():
    """Verifica robustez das métricas sob condições de contorno (N0 próximo ou igual a K)."""
    m = calcular_metricas_cineticas(
        r_sac=0.23, k_sac=10.0, n0_sac=10.0,
        r_sch=0.06, k_sch=5.0, n0_sch=5.0
    )
    assert m['t_inflex_sac'] == 0.0
    assert m['t_inflex_sch'] == 0.0
    assert m['td_sac'] > 0
    assert m['td_sch'] > 0
