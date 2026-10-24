"""
=============================================================================
Callbacks da Aplicação: Modelo de Crescimento Logístico (Gause, 1932)
=============================================================================
"""

import dash
from dash import Input, Output
import numpy as np
import plotly.graph_objects as go

from src.model import (
    T_SAC_REAL,
    SAC_REAL,
    T_SCH_REAL,
    SCH_REAL,
    simular_logistica,
)


def register_callbacks(app):
    """
    Registra todos os callbacks reativos no app Dash.
    """

    # =========================================================================
    # CALLBACK 1: ATUALIZAÇÃO DOS BADGES DINÂMICOS DOS SLIDERS
    # =========================================================================
    @app.callback(
        [
            Output("badge_r_sac", "children"),
            Output("badge_k_sac", "children"),
            Output("badge_n0_sac", "children"),
            Output("badge_r_sch", "children"),
            Output("badge_k_sch", "children"),
            Output("badge_n0_sch", "children"),
            Output("badge_t_max", "children"),
        ],
        [
            Input("r_sac", "value"),
            Input("k_sac", "value"),
            Input("n0_sac", "value"),
            Input("r_sch", "value"),
            Input("k_sch", "value"),
            Input("n0_sch", "value"),
            Input("t_max", "value"),
        ]
    )
    def update_badges(r_sac, k_sac, n0_sac, r_sch, k_sch, n0_sch, t_max):
        b_r_sac = f"{r_sac:.2f} h⁻¹"
        b_k_sac = f"{k_sac:.1f} vol"
        b_n0_sac = f"{n0_sac:.2f} vol"
        
        b_r_sch = f"{r_sch:.3f} h⁻¹"
        b_k_sch = f"{k_sch:.1f} vol"
        b_n0_sch = f"{n0_sch:.2f} vol"
        
        b_t_max = f"{t_max} h"
        
        return b_r_sac, b_k_sac, b_n0_sac, b_r_sch, b_k_sch, b_n0_sch, b_t_max

    # =========================================================================
    # CALLBACK 2: SIMULAÇÃO, CÁLCULO DE KPIS E RENDERIZAÇÃO DO GRÁFICO
    # =========================================================================
    @app.callback(
        [
            Output('kpi_ksac_val', 'children'),
            Output('kpi_ksac_sub', 'children'),
            Output('kpi_rsac_val', 'children'),
            Output('kpi_rsac_sub', 'children'),
            Output('kpi_ksch_val', 'children'),
            Output('kpi_ksch_sub', 'children'),
            Output('kpi_fit_val', 'children'),
            Output('kpi_fit_sub', 'children'),
            Output('population_chart', 'figure'),
        ],
        [
            Input('r_sac', 'value'),
            Input('k_sac', 'value'),
            Input('n0_sac', 'value'),
            Input('r_sch', 'value'),
            Input('k_sch', 'value'),
            Input('n0_sch', 'value'),
            Input('t_max', 'value'),
            Input('filtro_especie', 'value'),
            Input('opcoes_grafico', 'value'),
            Input('modo_grafico', 'value'),
        ]
    )
    def simula_e_atualiza(r_sac, k_sac, n0_sac, r_sch, k_sch, n0_sch, t_max, filtro_especie, opcoes_grafico, modo_grafico):
        res = simular_logistica(
            r_sac=r_sac,
            k_sac=k_sac,
            n0_sac=n0_sac,
            r_sch=r_sch,
            k_sch=k_sch,
            n0_sch=n0_sch,
            t_max=t_max,
            n_points=1200
        )

        t_eval = res["t"]
        t_end = res["t_end"]
        sac_eval = res["sac"]
        sch_eval = res["sch"]
        dsac_dt_eval = res["dsac_dt"]
        dsch_dt_eval = res["dsch_dt"]

        k_sac_val = res["k_sac"]
        r_sac_val = res["r_sac"]
        td_sac = res["td_sac"]
        t_inflex_sac = res["t_inflex_sac"]
        max_taxa_sac = res["max_taxa_sac"]

        k_sch_val = res["k_sch"]
        r_sch_val = res["r_sch"]
        td_sch = res["td_sch"]
        t_inflex_sch = res["t_inflex_sch"]
        max_taxa_sch = res["max_taxa_sch"]

        r2_sac = res["r2_sac"]
        r2_sch = res["r2_sch"]
        rmse_sac = res["rmse_sac"]
        rmse_sch = res["rmse_sch"]

        # Formatação dos cards de KPIs
        kpi_ksac_val = f"{k_sac_val:.1f} vol"
        kpi_ksac_sub = f"Platô Gause: ~12.8 vol (K/2: {k_sac_val/2:.1f})"

        kpi_rsac_val = f"{r_sac_val:.2f} h⁻¹"
        kpi_rsac_sub = f"td ≈ {td_sac:.1f} h | Inflexão: {t_inflex_sac:.1f} h"

        kpi_ksch_val = f"{k_sch_val:.1f} vol"
        kpi_ksch_sub = f"Platô Gause: ~5.8 vol (K/2: {k_sch_val/2:.1f})"

        kpi_fit_val = f"{r2_sac:.2f} | {r2_sch:.2f}"
        kpi_fit_sub = f"RMSE Sac: {rmse_sac:.2f} | Sch: {rmse_sch:.2f}"

        # Construção da figura interativa com Plotly
        fig = go.Figure()

        opcoes = opcoes_grafico if opcoes_grafico is not None else []
        mostrar_dados = "dados_reais" in opcoes
        mostrar_linhas_k = "linhas_k" in opcoes
        mostrar_inflexao = "inflexao" in opcoes

        exibir_sac = filtro_especie in ["ambas", "sac"]
        exibir_sch = filtro_especie in ["ambas", "sch"]

        # =====================================================================
        # CASO 1: POPULAÇÃO N(t)
        # =====================================================================
        if modo_grafico == "pop":
            y_axis_title = "Biomassa / Volume Relativo de Levedura"
            
            # 1. Saccharomyces cerevisiae
            if exibir_sac:
                fig.add_trace(go.Scatter(
                    x=t_eval,
                    y=sac_eval,
                    name="S. cerevisiae (Modelo)",
                    line=dict(color="#1d4ed8", width=3.5),
                    hovertemplate="Tempo: %{x:.1f} h<br>Volume Sac: %{y:.2f} vol<extra></extra>"
                ))
                
                if mostrar_dados:
                    fig.add_trace(go.Scatter(
                        x=T_SAC_REAL,
                        y=SAC_REAL,
                        name="S. cerevisiae (Gause, 1932)",
                        mode="markers",
                        marker=dict(color="#1e40af", size=8, symbol="cross", line=dict(width=1.5, color="#ffffff")),
                        hovertemplate="Dado Real Sac: %{y:.2f} vol em %{x:.1f} h<extra></extra>"
                    ))
                
                if mostrar_linhas_k:
                    fig.add_hline(
                        y=k_sac_val,
                        line_dash="dot",
                        line_color="#3b82f6",
                        line_width=1.5,
                        annotation_text=f"K_sac = {k_sac_val:.1f}",
                        annotation_position="bottom right",
                        annotation_font=dict(size=10, color="#1d4ed8")
                    )
                
                if mostrar_inflexao and t_inflex_sac > 0 and t_inflex_sac <= t_end:
                    fig.add_trace(go.Scatter(
                        x=[t_inflex_sac],
                        y=[k_sac_val / 2.0],
                        name="Inflexão Sac (K/2)",
                        mode="markers+text",
                        marker=dict(color="#2563eb", size=10, symbol="diamond"),
                        text=[f"Inflexão Sac ({t_inflex_sac:.1f}h, {k_sac_val/2:.1f})"],
                        textposition="top center",
                        textfont=dict(size=10, color="#1d4ed8"),
                        hovertemplate="Inflexão Sac: %{y:.2f} vol em %{x:.1f} h<extra></extra>"
                    ))

            # 2. Schizosaccharomyces kephir
            if exibir_sch:
                fig.add_trace(go.Scatter(
                    x=t_eval,
                    y=sch_eval,
                    name="Sch. kephir (Modelo)",
                    line=dict(color="#dc2626", width=3.5),
                    hovertemplate="Tempo: %{x:.1f} h<br>Volume Sch: %{y:.2f} vol<extra></extra>"
                ))
                
                if mostrar_dados:
                    fig.add_trace(go.Scatter(
                        x=T_SCH_REAL,
                        y=SCH_REAL,
                        name="Sch. kephir (Gause, 1932)",
                        mode="markers",
                        marker=dict(color="#b91c1c", size=8, symbol="circle", line=dict(width=1.5, color="#ffffff")),
                        hovertemplate="Dado Real Sch: %{y:.2f} vol em %{x:.1f} h<extra></extra>"
                    ))
                
                if mostrar_linhas_k:
                    fig.add_hline(
                        y=k_sch_val,
                        line_dash="dash",
                        line_color="#f87171",
                        line_width=1.5,
                        annotation_text=f"K_sch = {k_sch_val:.1f}",
                        annotation_position="bottom right",
                        annotation_font=dict(size=10, color="#b91c1c")
                    )
                
                if mostrar_inflexao and t_inflex_sch > 0 and t_inflex_sch <= t_end:
                    fig.add_trace(go.Scatter(
                        x=[t_inflex_sch],
                        y=[k_sch_val / 2.0],
                        name="Inflexão Sch (K/2)",
                        mode="markers+text",
                        marker=dict(color="#dc2626", size=10, symbol="diamond"),
                        text=[f"Inflexão Sch ({t_inflex_sch:.1f}h, {k_sch_val/2:.1f})"],
                        textposition="top center",
                        textfont=dict(size=10, color="#b91c1c"),
                        hovertemplate="Inflexão Sch: %{y:.2f} vol em %{x:.1f} h<extra></extra>"
                    ))

            max_y = max(
                k_sac_val * 1.15 if exibir_sac else 0.0,
                k_sch_val * 1.15 if exibir_sch else 0.0,
                14.5
            )
            y_range = [0, max_y]

        # =====================================================================
        # CASO 2: TAXA DE CRESCIMENTO dN/dt
        # =====================================================================
        else:
            y_axis_title = "Taxa de Variação dN/dt (vol / hora)"
            
            if exibir_sac:
                fig.add_trace(go.Scatter(
                    x=t_eval,
                    y=dsac_dt_eval,
                    name="Taxa dN/dt (S. cerevisiae)",
                    line=dict(color="#1d4ed8", width=3.5),
                    hovertemplate="Tempo: %{x:.1f} h<br>Taxa Sac: %{y:.3f} vol/h<extra></extra>"
                ))
                
                if t_inflex_sac > 0 and t_inflex_sac <= t_end:
                    fig.add_trace(go.Scatter(
                        x=[t_inflex_sac],
                        y=[max_taxa_sac],
                        name="Pico Máximo dN/dt Sac",
                        mode="markers+text",
                        marker=dict(color="#1d4ed8", size=10, symbol="star"),
                        text=[f"Pico Sac: {max_taxa_sac:.2f} vol/h ({t_inflex_sac:.1f}h)"],
                        textposition="top center",
                        textfont=dict(size=10, color="#1d4ed8"),
                        hovertemplate="Pico Taxa Sac: %{y:.3f} vol/h em %{x:.1f} h<extra></extra>"
                    ))

            if exibir_sch:
                fig.add_trace(go.Scatter(
                    x=t_eval,
                    y=dsch_dt_eval,
                    name="Taxa dN/dt (Sch. kephir)",
                    line=dict(color="#dc2626", width=3.5),
                    hovertemplate="Tempo: %{x:.1f} h<br>Taxa Sch: %{y:.3f} vol/h<extra></extra>"
                ))
                
                if t_inflex_sch > 0 and t_inflex_sch <= t_end:
                    fig.add_trace(go.Scatter(
                        x=[t_inflex_sch],
                        y=[max_taxa_sch],
                        name="Pico Máximo dN/dt Sch",
                        mode="markers+text",
                        marker=dict(color="#dc2626", size=10, symbol="star"),
                        text=[f"Pico Sch: {max_taxa_sch:.2f} vol/h ({t_inflex_sch:.1f}h)"],
                        textposition="top center",
                        textfont=dict(size=10, color="#dc2626"),
                        hovertemplate="Pico Taxa Sch: %{y:.3f} vol/h em %{x:.1f} h<extra></extra>"
                    ))

            max_deriv = max(
                np.max(dsac_dt_eval) * 1.25 if exibir_sac else 0.0,
                np.max(dsch_dt_eval) * 1.25 if exibir_sch else 0.0,
                0.5
            )
            y_range = [0, max_deriv]

        # Configuração de layout e estilo visual idêntico ao padrão dos dashboards
        fig.update_layout(
            template="plotly_white",
            title=dict(
                text="<b>Curvas de Crescimento e Ajuste aos Dados de Gause (1932)</b>" if modo_grafico == "pop" else "<b>Taxa Instantânea de Variação Populacional (dN/dt)</b>",
                font=dict(size=14, color="#1e293b"),
                x=0.02,
                xanchor="left",
                y=0.96,
                yref="container",
                yanchor="top"
            ),
            xaxis=dict(
                title="Tempo (horas)",
                range=[0, t_end],
                gridcolor="#f1f5f9",
                zerolinecolor="#cbd5e1"
            ),
            yaxis=dict(
                title=y_axis_title,
                range=y_range,
                gridcolor="#f1f5f9",
                zerolinecolor="#cbd5e1"
            ),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.18,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#e2e8f0",
                borderwidth=1
            ),
            margin=dict(l=55, r=30, t=55, b=85),
            hovermode="x unified"
        )

        return (
            kpi_ksac_val, kpi_ksac_sub,
            kpi_rsac_val, kpi_rsac_sub,
            kpi_ksch_val, kpi_ksch_sub,
            kpi_fit_val, kpi_fit_sub,
            fig
        )
