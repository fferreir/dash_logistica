"""
=============================================================================
Layout da Aplicação: Modelo de Crescimento Logístico (Gause, 1932)
=============================================================================
"""

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc


# =============================================================================
# CABEÇALHO HERO
# =============================================================================
cabecalho = html.Div(
    [
        html.Div(
            [
                html.H1(
                    [
                        html.I(className="fa-solid fa-chart-line me-2"),
                        "Modelo de Crescimento Logístico de Populações"
                    ],
                    className="display-6 mb-1"
                ),
                html.P(
                    "Simulador dinâmico baseado nos experimentos clássicos de G. F. Gause (1932) com Saccharomyces cerevisiae e Schizosaccharomyces kephir.",
                    className="lead fs-6 mb-0 opacity-75"
                ),
            ],
            className="hero-header p-3 mb-3"
        )
    ]
)


# =============================================================================
# CARDS DE MÉTRICAS (KPIS)
# =============================================================================
cards_metricas = dbc.Row(
    [
        # KPI 1: Saccharomyces K
        dbc.Col(
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Capacidade K (S. cerevisiae)", className="kpi-title"),
                            html.Div(id="kpi_ksac_val", className="kpi-value"),
                            html.Div(id="kpi_ksac_sub", className="kpi-subtitle"),
                            html.I(className="fa-solid fa-flask kpi-icon"),
                        ],
                        className="kpi-card-inner"
                    )
                ],
                className="kpi-card kpi-sac"
            ),
            width=6, sm=6, lg=3, className="mb-2"
        ),
        # KPI 2: Saccharomyces Cinética
        dbc.Col(
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Cinética (S. cerevisiae)", className="kpi-title"),
                            html.Div(id="kpi_rsac_val", className="kpi-value"),
                            html.Div(id="kpi_rsac_sub", className="kpi-subtitle"),
                            html.I(className="fa-solid fa-stopwatch kpi-icon"),
                        ],
                        className="kpi-card-inner"
                    )
                ],
                className="kpi-card kpi-rate"
            ),
            width=6, sm=6, lg=3, className="mb-2"
        ),
        # KPI 3: Schizosaccharomyces K
        dbc.Col(
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Capacidade K (Sch. kephir)", className="kpi-title"),
                            html.Div(id="kpi_ksch_val", className="kpi-value"),
                            html.Div(id="kpi_ksch_sub", className="kpi-subtitle"),
                            html.I(className="fa-solid fa-vial kpi-icon"),
                        ],
                        className="kpi-card-inner"
                    )
                ],
                className="kpi-card kpi-sch"
            ),
            width=6, sm=6, lg=3, className="mb-2"
        ),
        # KPI 4: Qualidade do Ajuste (R² e RMSE)
        dbc.Col(
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Qualidade do Ajuste (R²)", className="kpi-title"),
                            html.Div(id="kpi_fit_val", className="kpi-value"),
                            html.Div(id="kpi_fit_sub", className="kpi-subtitle"),
                            html.I(className="fa-solid fa-bullseye kpi-icon"),
                        ],
                        className="kpi-card-inner"
                    )
                ],
                className="kpi-card kpi-fit"
            ),
            width=6, sm=6, lg=3, className="mb-2"
        ),
    ],
    className="mb-2"
)


# =============================================================================
# DOCUMENTAÇÃO TEÓRICA E PERGUNTAS PEDAGÓGICAS (ACCORDION)
# =============================================================================
descricao = html.Div([
    dcc.Markdown(
        r'''
        O **modelo de crescimento logístico** foi proposto originalmente pelo matemático belga **Pierre François Verhulst** (1838) e redescoberto independentemente por **Raymond Pearl e Lowell Reed** (1920). 

        Diferente do crescimento exponencial ilimitado ($$\frac{dN}{dt} = rN$$), o modelo logístico incorpora a noção ecológica de que o ambiente possui recursos finitos (espaço, substrato, nutrientes), impondo uma **capacidade de suporte ($$K$$)**.

        Em 1932, o biólogo soviético **Georgyi Frantsevich Gause** realizou uma célebre série de experimentos com leveduras cultivadas em tubos de ensaio com meio líquido padronizado, quantificando o volume de células ao longo do tempo (em horas) e fornecendo uma das primeiras validações empíricas rigorosas da equação logística.
        ''', mathjax=True
    ),
    dcc.Markdown(
        r'''
        ### Formulação Matemática:
        $$\frac{dN}{dt} = r N \left(1 - \frac{N}{K}\right)$$

        A solução analítica obtida por integração de variáveis separáveis e frações parciais com a condição inicial $$N(0) = N_0$$ é:
        $$N(t) = \frac{K}{1 + \left(\frac{K - N_0}{N_0}\right) e^{-r t}}$$

        ### Propriedades Notáveis da Curva Sigmoide:
        * **Fase inicial ($$N \ll K$$):** O termo $$\left(1 - \frac{N}{K}\right) \approx 1$$, e a população cresce quasi-exponencialmente com taxa $$r$$.
        * **Ponto de Inflexão ($$N = K/2$$):** Onde a derivada segunda $$\frac{d^2N}{dt^2} = 0$$. É o instante de taxa de crescimento absoluto máxima:
          $$\left(\frac{dN}{dt}\right)_{\max} = \frac{r K}{4}, \quad t^* = \frac{1}{r} \ln\left(\frac{K - N_0}{N_0}\right)$$
        * **Fase de Saturação ($$t \to \infty$$):** A população aproxima-se assintoticamente do limite de suporte $$K$$, com $$\frac{dN}{dt} \to 0$$.
        ''', mathjax=True
    )
])

parametros = dcc.Markdown(
    r'''
    * **$$r$$ (Taxa intrínseca de crescimento per capita):** Expressa em $$h^{-1}$$. Mede a capacidade inata de proliferação populacional na ausência de limitações de recursos.
    * **$$K$$ (Capacidade de suporte):** Expressa na mesma unidade da população (unidades de volume de biomassa de Gause). Representa a densidade populacional máxima sustentável pelo meio de cultivo.
    * **$$N_0$$ (Densidade do inóculo inicial):** Quantidade inicial de biomassa de leveduras inoculada no instante $$t = 0$$.
    * **$$t_d$$ (Tempo de duplicação exponencial inicial):** Tempo necessário para a população duplicar de tamanho na fase de crescimento livre:
      $$t_d = \frac{\ln(2)}{r} \approx \frac{0.6931}{r} \text{ horas}$$
    * **$$t^*$$ (Tempo até o ponto de inflexão):** Instante em que a população atinge metade da capacidade de suporte ($$N = K/2$$).
    * **$$R^2$$ (Coeficiente de Determinação):** Mede a proporção da variância dos dados experimentais de Gause explicada pelo modelo calibrado ($$1.0$$ indica ajuste perfeito).
    * **RMSE (Raiz do Erro Quadrático Médio):** Erro médio de predição na mesma unidade de medida da população.
    ''', mathjax=True
)

cond_inicial = html.Div([
    dcc.Markdown(
        r'''
        Os dados empíricos originais foram publicados em:
        > **GAUSE, G. F.** *Experimental Studies on the Struggle for Existence: 1. Mixed Population of Two Species of Yeast.* **Journal of Experimental Biology**, v. 9, n. 4, p. 389–402, 1932.

        ### Resumo das Culturas em Monocultura:
        * **Saccharomyces cerevisiae:** Cultivada em meio padronizado a 28°C. Apresenta multiplicação rápida por brotamento celular, atingindo saturação em cerca de 40 a 50 horas, com capacidade suporte observada $$K \approx 12.8$$ a $$13.0$$ e taxa $$r \approx 0.22 - 0.26 \text{ h}^{-1}$$.
        * **Schizosaccharomyces kephir:** Levedura de fissão binária mais lenta. Requer mais de 100 horas para atingir o platô assintótico, estabilizando em $$K \approx 5.8$$ com taxa $$r \approx 0.057 - 0.060 \text{ h}^{-1}$$.
        ''', mathjax=True
    ),
    dbc.Table(
        [
            html.Thead(
                html.Tr([
                    html.Th("Espécie"),
                    html.Th("Inóculo N₀"),
                    html.Th("Capacidade Suporte K"),
                    html.Th("Taxa Intrínseca r"),
                    html.Th("Tempo de Duplicação td"),
                    html.Th("Morfologia de Divisão")
                ])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(html.I("Saccharomyces cerevisiae")),
                    html.Td("≈ 0.45 vol"),
                    html.Td("≈ 12.8 - 13.0 vol"),
                    html.Td("≈ 0.22 - 0.26 h⁻¹"),
                    html.Td("≈ 2.7 - 3.1 horas"),
                    html.Td("Brotamento celular (gemação)")
                ]),
                html.Tr([
                    html.Td(html.I("Schizosaccharomyces kephir")),
                    html.Td("≈ 0.46 vol"),
                    html.Td("≈ 5.8 vol"),
                    html.Td("≈ 0.057 - 0.060 h⁻¹"),
                    html.Td("≈ 11.5 - 12.0 horas"),
                    html.Td("Fissão binária simétrica")
                ]),
            ])
        ],
        bordered=True,
        hover=True,
        size="sm",
        className="table-data mt-2"
    )
])

perguntas = dcc.Markdown(
    r'''
    1. **Identificação e Significado da Capacidade de Suporte ($$K$$):**
       Observe o comportamento assintótico dos dados experimentais nas horas finais ($$t > 35\text{ h}$$ para *Saccharomyces* e $$t > 90\text{ h}$$ para *Schizosaccharomyces*). Qual valor de $$K$$ melhor representa o platô estacionário de cada espécie? Por que a capacidade de suporte de *Saccharomyces* ($$\approx 12.8$$) é mais que o dobro da de *Schizosaccharomyces* ($$\approx 5.8$$), mesmo tendo sido cultivadas no mesmo volume de meio nutritivo? Quais fatores intrínsecos de eficiência na conversão de glicose e tolerância metabólica explicam essa diferença?
    
    2. **Taxa Intrínseca ($$r$$) e Tempo de Duplicação ($$t_d$$):**
       Ajuste a taxa $$r$$ para ambas as espécies. Qual das duas leveduras apresenta maior velocidade de proliferação na fase inicial (exponencial)? Calcule o tempo de duplicação teórico ($$t_d = \frac{\ln 2}{r}$$) de cada espécie. Como o modo de divisão celular de cada organismo — brotamento assimétrico com descolamento de gemas em *S. cerevisiae* versus fissão binária simétrica em *Sch. kephir* — ajuda a justificar essa disparidade temporal?
    
    3. **Ponto de Inflexão e Taxa Máxima de Crescimento ($$\frac{dN}{dt}$$):**
       Habilite a visualização do ponto de inflexão ($$N = K/2$$) e alterne a escala do gráfico para *"Taxa de Crescimento dN/dt"*. Em que instante temporal ($$t^*$$) cada população atinge seu pico máximo de produtividade populacional? Por que, biologicamente, a velocidade de acréscimo populacional começa a frear logo após a população ultrapassar a metade de $$K$$, mesmo havendo ainda substrato disponível no meio?
    
    4. **Sensibilidade ao Inóculo Inicial ($$N_0$$):**
       Varie a densidade inicial $$N_0$$ nos controles. O que acontece com a curva se o inóculo for extremamente reduzido ($$N_0 \approx 0.1$$)? E se você simular uma cultura iniciada com supersaturação, isto é, com densidade inicial superior à capacidade de suporte ($$N_0 > K$$)? Como a equação logística responde matematicamente e qual seria o significado biológico de uma população decrescendo em direção a $$K$$?
    
    5. **Aderência aos Dados ($$R^2$$, RMSE) e Limitações do Modelo:**
       Acompanhe os indicadores de qualidade de ajuste nos cards de métricas superiores. Qual conjunto de parâmetros ($$r, K, N_0$$) fornece o maior $$R^2$$ e o menor RMSE? Analise criticamente os primeiros pontos experimentais ($$t < 15\text{ h}$$): o modelo logístico reproduz a fase de adaptação inicial (*lag phase*) observada em leveduras? Por que em culturas biológicas fechadas reais em batelada a população não permanece indefinidamente no platô $$K$$, sofrendo posterior declínio por depleção de açúcares e acúmulo de etanol?
    ''', mathjax=True
)

textos_descricao = html.Div(
    dbc.Accordion(
        [
            dbc.AccordionItem(descricao, title="📘 Descrição e Formulação do Modelo"),
            dbc.AccordionItem(parametros, title="📊 Parâmetros do Modelo"),
            dbc.AccordionItem(cond_inicial, title="🔬 Dados Experimentais de Gause (1932)"),
            dbc.AccordionItem(perguntas, title="❓ Questões para Estudo e Exploração"),
        ],
        start_collapsed=True,
        flush=False,
    )
)


# =============================================================================
# COMPONENTES DE CONTROLE (CARDS COM SLIDERS E BADGES)
# =============================================================================
ajuste_saccharomyces = html.Div(
    [
        html.Div(
            [
                html.I(className="fa-solid fa-flask me-2 text-primary"),
                "Parâmetros: Saccharomyces cerevisiae"
            ],
            className="card-header"
        ),
        html.Div(
            [
                html.Div([
                    dbc.Label(dcc.Markdown(r'Taxa de crescimento ($$r_{sac}$$):', mathjax=True), className="mb-0"),
                    html.Span(id="badge_r_sac", className="slider-val-badge")
                ], className="slider-label"),
                dcc.Slider(
                    id="r_sac",
                    min=0.01,
                    max=0.60,
                    step=0.01,
                    value=0.23,
                    tooltip={"placement": "bottom", "always_visible": False}
                ),
            ],
            className="slider-group",
        ),
        html.Div(
            [
                html.Div([
                    dbc.Label(dcc.Markdown(r'Capacidade de suporte ($$K_{sac}$$):', mathjax=True), className="mb-0"),
                    html.Span(id="badge_k_sac", className="slider-val-badge")
                ], className="slider-label"),
                dcc.Slider(
                    id="k_sac",
                    min=1.0,
                    max=25.0,
                    step=0.1,
                    value=13.0,
                    tooltip={"placement": "bottom", "always_visible": False}
                ),
            ],
            className="slider-group",
        ),
        html.Div(
            [
                html.Div([
                    dbc.Label(dcc.Markdown(r'Inóculo inicial ($$N_{0,sac}$$):', mathjax=True), className="mb-0"),
                    html.Span(id="badge_n0_sac", className="slider-val-badge")
                ], className="slider-label"),
                dcc.Slider(
                    id="n0_sac",
                    min=0.05,
                    max=3.00,
                    step=0.05,
                    value=0.45,
                    tooltip={"placement": "bottom", "always_visible": False}
                ),
            ],
            className="slider-group",
        ),
    ],
    className="card-control mb-3",
)

ajuste_schizosaccharomyces = html.Div(
    [
        html.Div(
            [
                html.I(className="fa-solid fa-vial me-2 text-danger"),
                "Parâmetros: Schizosaccharomyces kephir"
            ],
            className="card-header"
        ),
        html.Div(
            [
                html.Div([
                    dbc.Label(dcc.Markdown(r'Taxa de crescimento ($$r_{sch}$$):', mathjax=True), className="mb-0"),
                    html.Span(id="badge_r_sch", className="slider-val-badge badge-sch")
                ], className="slider-label"),
                dcc.Slider(
                    id="r_sch",
                    min=0.005,
                    max=0.300,
                    step=0.005,
                    value=0.060,
                    tooltip={"placement": "bottom", "always_visible": False}
                ),
            ],
            className="slider-group",
        ),
        html.Div(
            [
                html.Div([
                    dbc.Label(dcc.Markdown(r'Capacidade de suporte ($$K_{sch}$$):', mathjax=True), className="mb-0"),
                    html.Span(id="badge_k_sch", className="slider-val-badge badge-sch")
                ], className="slider-label"),
                dcc.Slider(
                    id="k_sch",
                    min=1.0,
                    max=15.0,
                    step=0.1,
                    value=5.8,
                    tooltip={"placement": "bottom", "always_visible": False}
                ),
            ],
            className="slider-group",
        ),
        html.Div(
            [
                html.Div([
                    dbc.Label(dcc.Markdown(r'Inóculo inicial ($$N_{0,sch}$$):', mathjax=True), className="mb-0"),
                    html.Span(id="badge_n0_sch", className="slider-val-badge badge-sch")
                ], className="slider-label"),
                dcc.Slider(
                    id="n0_sch",
                    min=0.05,
                    max=3.00,
                    step=0.05,
                    value=0.46,
                    tooltip={"placement": "bottom", "always_visible": False}
                ),
            ],
            className="slider-group",
        ),
    ],
    className="card-control mb-3",
)

ajuste_opcoes = html.Div(
    [
        html.Div(
            [
                html.I(className="fa-solid fa-sliders me-2 text-secondary"),
                "Opções de Simulação e Visualização"
            ],
            className="card-header"
        ),
        html.Div(
            [
                html.Div([
                    dbc.Label("Horizonte de Tempo (horas)", className="mb-0"),
                    html.Span(id="badge_t_max", className="slider-val-badge badge-neutral")
                ], className="slider-label"),
                dcc.Slider(
                    id="t_max",
                    min=40,
                    max=200,
                    step=10,
                    value=150,
                    tooltip={"placement": "bottom", "always_visible": False}
                ),
            ],
            className="slider-group",
        ),
        html.Div(
            [
                dbc.Label("Espécies Exibidas no Gráfico:", className="fw-semibold small text-muted mb-1"),
                dbc.RadioItems(
                    id="filtro_especie",
                    options=[
                        {"label": " Ambas as espécies", "value": "ambas"},
                        {"label": " Apenas Saccharomyces", "value": "sac"},
                        {"label": " Apenas Schizosaccharomyces", "value": "sch"},
                    ],
                    value="ambas",
                    className="small"
                ),
            ],
            className="slider-group",
        ),
        html.Div(
            [
                dbc.Label("Elementos de Referência:", className="fw-semibold small text-muted mb-1"),
                dbc.Checklist(
                    id="opcoes_grafico",
                    options=[
                        {"label": " Dados reais de Gause (1932)", "value": "dados_reais"},
                        {"label": " Linhas de Capacidade de Suporte (K)", "value": "linhas_k"},
                        {"label": " Pontos de Inflexão (K/2)", "value": "inflexao"},
                    ],
                    value=["dados_reais", "linhas_k"],
                    className="small"
                ),
            ],
            className="slider-group",
        ),
    ],
    className="card-control mb-3",
)


# =============================================================================
# RODAPÉ INSTITUCIONAL (FOOTER COM LINK PARA O GITHUB)
# =============================================================================
rodape = html.Footer(
    html.Div(
        [
            html.Small(
                [
                    "Simulador do Modelo de Crescimento Logístico (Gause, 1932) • Desenvolvido com Dash, Plotly, SciPy e Bootstrap • ",
                    html.A(
                        [
                            html.I(className="fa-brands fa-github me-1"),
                            "Código Aberto no GitHub"
                        ],
                        href="https://github.com/fferreir/dash_logistica",
                        target="_blank",
                        className="text-decoration-none text-muted fw-semibold"
                    )
                ],
                className="text-muted"
            )
        ],
        className="text-center py-3 mt-4 border-top"
    )
)


# =============================================================================
# FUNÇÃO PRINCIPAL DE CRIAÇÃO DO LAYOUT
# =============================================================================
def create_layout():
    """
    Constrói a árvore de componentes da interface gráfica do dashboard.
    """
    return dbc.Container(
        [
            cabecalho,
            dbc.Row(
                [
                    # Coluna Esquerda: Controles dos Parâmetros e Opções (4 colunas em lg)
                    dbc.Col(
                        [
                            ajuste_saccharomyces,
                            ajuste_schizosaccharomyces,
                            ajuste_opcoes,
                        ],
                        width=12, lg=4
                    ),
                    
                    # Coluna Direita: Cards de Métricas, Gráfico e Accordion Teórico (8 colunas em lg)
                    dbc.Col(
                        [
                            # Linha com os 4 Cards de Métricas (KPIs)
                            cards_metricas,
                            
                            # Card do Gráfico com Seletor de Variável
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.I(className="fa-solid fa-chart-area me-2 text-primary"),
                                                    html.Span(
                                                        "Dinâmica do Crescimento Logístico de Leveduras",
                                                        className="fw-bold text-dark me-3"
                                                    ),
                                                ],
                                                className="d-flex align-items-center mb-1 mb-md-0"
                                            ),
                                            html.Div(
                                                [
                                                    dbc.Label("Variável:", className="me-2 mb-0 small text-muted fw-semibold"),
                                                    dbc.RadioItems(
                                                        id="modo_grafico",
                                                        options=[
                                                            {"label": " População N(t)", "value": "pop"},
                                                            {"label": " Taxa de Crescimento dN/dt", "value": "derivada"},
                                                        ],
                                                        value="pop",
                                                        inline=True,
                                                        className="small"
                                                    ),
                                                ],
                                                className="d-flex align-items-center"
                                            )
                                        ],
                                        className="d-flex flex-wrap justify-content-between align-items-center mb-2 px-1"
                                    ),
                                    dcc.Loading(
                                        dcc.Graph(
                                            id='population_chart',
                                            style={'height': '530px'},
                                            config={
                                                "displayModeBar": True,
                                                "responsive": True,
                                                "modeBarButtonsToRemove": ["lasso2d", "select2d"]
                                            }
                                        ),
                                        type="circle"
                                    )
                                ],
                                className="graph-card mb-3"
                            ),
                            
                            # Accordion Teórico e Perguntas logo abaixo do gráfico
                            html.Div(textos_descricao, className="mb-4")
                        ],
                        width=12, lg=8
                    ),
                ],
                className="g-3"
            ),
            rodape,
        ],
        fluid=True,
        className="py-2 px-3"
    )
