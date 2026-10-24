import os
import flask
import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go

# =============================================================================
# CONFIGURAÇÃO DE ROTA E PREFIXO (Suporte para dev local e proxy reverso de deploy)
# =============================================================================
prefix = os.getenv('DASH_REQUESTS_PATHNAME_PREFIX', '/dash_logistica/')
if not prefix.startswith('/'):
    prefix = '/' + prefix
if not prefix.endswith('/'):
    prefix = prefix + '/'

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.LUMEN, dbc.icons.FONT_AWESOME],
    requests_pathname_prefix=prefix,
    routes_pathname_prefix=prefix,
    suppress_callback_exceptions=True
)
server = app.server

# Redirecionamento da raiz '/' para o prefixo da aplicação em desenvolvimento local
if prefix != '/':
    @server.route('/')
    def root_redirect():
        return flask.redirect(prefix)


# =============================================================================
# DADOS EXPERIMENTAIS CLÁSSICOS DE G. F. GAUSE (1932)
# "Experimental Studies on the Struggle for Existence: 1. Mixed Population of Two Species of Yeast"
# Tempo medido em horas (h); população expressa em unidades de volume de levedura.
# =============================================================================
T_SAC_REAL = np.array([6.0, 7.5, 15.0, 16.0, 24.0, 24.0, 29.0, 31.5, 33.0, 40.0, 44.0, 48.0, 51.5, 53.0])
SAC_REAL = np.array([0.37, 1.63, 6.20, 8.87, 10.66, 10.97, 12.50, 12.60, 12.90, 13.27, 12.77, 12.87, 12.90, 12.70])

T_SCH_REAL = np.array([15.0, 16.0, 29.0, 31.5, 48.0, 51.5, 72.0, 93.0, 117.0, 141.0])
SCH_REAL = np.array([1.27, 1.00, 1.70, 2.33, 2.73, 4.56, 4.87, 5.67, 5.80, 5.83])


# =============================================================================
# SISTEMA DE EQUAÇÕES DIFERENCIAIS ORDINÁRIAS (MODELO LOGÍSTICO)
# =============================================================================
def ode_sys(t, state, r_sac, k_sac, r_sch, k_sch):
    """
    Sistema desacoplado de crescimento logístico (monoculturas de Gause, 1932).
    dN/dt = r * N * (1 - N / K)
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
    
    termo_exp = ((k_eff - n0_eff) / n0_eff) * np.exp(-r_eff * t_arr)
    return k_eff / (1.0 + termo_exp)


# =============================================================================
# CABEÇALHO DO DASHBOARD (HERO HEADER)
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
# LAYOUT PRINCIPAL DO DASHBOARD
# =============================================================================
app.layout = dbc.Container(
    [
        cabecalho,
        dbc.Row(
            [
                # Coluna Esquerda: Controles dos Parâmetros e Opções (4 colunas)
                dbc.Col(
                    [
                        ajuste_saccharomyces,
                        ajuste_schizosaccharomyces,
                        ajuste_opcoes,
                    ],
                    width=12, lg=4
                ),
                
                # Coluna Direita: Cards de Métricas, Gráfico e Accordion Teórico (8 colunas)
                dbc.Col(
                    [
                        # Linha com os 4 Cards de Métricas (KPIs)
                        cards_metricas,
                        
                        # Card do Gráfico com Seletor de Modo
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
    ],
    fluid=True,
    className="py-2 px-3"
)


# =============================================================================
# CALLBACK DE ATUALIZAÇÃO DOS BADGES DOS SLIDERS
# =============================================================================
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


# =============================================================================
# CALLBACK PRINCIPAL: SIMULAÇÃO, CÁLCULO DE KPIS E RENDERIZAÇÃO DO GRÁFICO
# =============================================================================
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
    # Proteção de conversão segura para números válidos
    r_sac_val = float(r_sac) if r_sac is not None else 0.23
    k_sac_val = float(k_sac) if k_sac is not None else 13.0
    n0_sac_val = float(n0_sac) if n0_sac is not None else 0.45

    r_sch_val = float(r_sch) if r_sch is not None else 0.060
    k_sch_val = float(k_sch) if k_sch is not None else 5.8
    n0_sch_val = float(n0_sch) if n0_sch is not None else 0.46

    t_end = float(t_max) if t_max is not None else 150.0
    t_eval = np.linspace(0.0, t_end, 1200)

    # Resolução analítica e por EDO
    sac_eval = solucao_analitica(t_eval, r_sac_val, k_sac_val, n0_sac_val)
    sch_eval = solucao_analitica(t_eval, r_sch_val, k_sch_val, n0_sch_val)

    # Derivadas de crescimento instantâneo (dN/dt)
    dsac_dt_eval = r_sac_val * sac_eval * (1.0 - (sac_eval / max(k_sac_val, 1e-4)))
    dsch_dt_eval = r_sch_val * sch_eval * (1.0 - (sch_eval / max(k_sch_val, 1e-4)))

    # Cálculos cinéticos
    td_sac = np.log(2.0) / max(r_sac_val, 1e-5)
    td_sch = np.log(2.0) / max(r_sch_val, 1e-5)

    # Tempo de inflexão
    if k_sac_val > n0_sac_val and n0_sac_val > 0:
        t_inflex_sac = (1.0 / max(r_sac_val, 1e-5)) * np.log((k_sac_val - n0_sac_val) / n0_sac_val)
    else:
        t_inflex_sac = 0.0

    if k_sch_val > n0_sch_val and n0_sch_val > 0:
        t_inflex_sch = (1.0 / max(r_sch_val, 1e-5)) * np.log((k_sch_val - n0_sch_val) / n0_sch_val)
    else:
        t_inflex_sch = 0.0

    # Avaliação da qualidade do ajuste (Goodness of Fit) contra dados reais de Gause
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

    # Construção dos KPIs formatados
    kpi_ksac_val = f"{k_sac_val:.1f} vol"
    kpi_ksac_sub = f"Platô Gause: ~12.8 vol (K/2: {k_sac_val/2:.1f})"

    kpi_rsac_val = f"{r_sac_val:.2f} h⁻¹"
    kpi_rsac_sub = f"td ≈ {td_sac:.1f} h | Inflexão: {t_inflex_sac:.1f} h"

    kpi_ksch_val = f"{k_sch_val:.1f} vol"
    kpi_ksch_sub = f"Platô Gause: ~5.8 vol (K/2: {k_sch_val/2:.1f})"

    kpi_fit_val = f"{r2_sac:.2f} | {r2_sch:.2f}"
    kpi_fit_sub = f"RMSE Sac: {rmse_sac:.2f} | Sch: {rmse_sch:.2f}"

    # Construção da figura com Plotly
    fig = go.Figure()

    opcoes = opcoes_grafico if opcoes_grafico is not None else []
    mostrar_dados = "dados_reais" in opcoes
    mostrar_linhas_k = "linhas_k" in opcoes
    mostrar_inflexao = "inflexao" in opcoes

    exibir_sac = filtro_especie in ["ambas", "sac"]
    exibir_sch = filtro_especie in ["ambas", "sch"]

    # ==========================
    # CASO 1: POPULAÇÃO N(t)
    # ==========================
    if modo_grafico == "pop":
        y_axis_title = "Biomassa / Volume Relativo de Levedura"
        
        # 1. Saccharomyces
        if exibir_sac:
            # Curva teórica
            fig.add_trace(go.Scatter(
                x=t_eval,
                y=sac_eval,
                name="S. cerevisiae (Modelo)",
                line=dict(color="#1d4ed8", width=3.5),
                hovertemplate="Tempo: %{x:.1f} h<br>Volume Sac: %{y:.2f} vol<extra></extra>"
            ))
            
            # Dados experimentais reais de Gause
            if mostrar_dados:
                fig.add_trace(go.Scatter(
                    x=T_SAC_REAL,
                    y=SAC_REAL,
                    name="S. cerevisiae (Gause, 1932)",
                    mode="markers",
                    marker=dict(color="#1e40af", size=8, symbol="cross", line=dict(width=1.5, color="#ffffff")),
                    hovertemplate="Dado Real Sac: %{y:.2f} vol em %{x:.1f} h<extra></extra>"
                ))
            
            # Linha de Capacidade K
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
            
            # Ponto de Inflexão (K/2)
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

        # 2. Schizosaccharomyces
        if exibir_sch:
            # Curva teórica
            fig.add_trace(go.Scatter(
                x=t_eval,
                y=sch_eval,
                name="Sch. kephir (Modelo)",
                line=dict(color="#dc2626", width=3.5),
                hovertemplate="Tempo: %{x:.1f} h<br>Volume Sch: %{y:.2f} vol<extra></extra>"
            ))
            
            # Dados experimentais reais de Gause
            if mostrar_dados:
                fig.add_trace(go.Scatter(
                    x=T_SCH_REAL,
                    y=SCH_REAL,
                    name="Sch. kephir (Gause, 1932)",
                    mode="markers",
                    marker=dict(color="#b91c1c", size=8, symbol="circle", line=dict(width=1.5, color="#ffffff")),
                    hovertemplate="Dado Real Sch: %{y:.2f} vol em %{x:.1f} h<extra></extra>"
                ))
            
            # Linha de Capacidade K
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
            
            # Ponto de Inflexão (K/2)
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

    # ==========================
    # CASO 2: TAXA DE CRESCIMENTO dN/dt
    # ==========================
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
            
            max_taxa_sac = (r_sac_val * k_sac_val) / 4.0
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
            
            max_taxa_sch = (r_sch_val * k_sch_val) / 4.0
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

    # Configuração de layout e estilo visual idêntico ao dash-sir
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


# =============================================================================
# EXECUÇÃO DO APLICATIVO
# =============================================================================
if __name__ == '__main__':
    port = int(os.getenv('PORT', 8050))
    app.run(debug=False, host='0.0.0.0', port=port)
