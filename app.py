import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from numpy import heaviside
from textwrap import dedent
import plotly.graph_objects as go


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUMEN, dbc.icons.FONT_AWESOME], requests_pathname_prefix='/dash_logistica/' )
server = app.server


cabecalho = html.H1("Modelo de Crescimento Logistico",className="bg-primary text-white p-2 mb-4")

descricao = dcc.Markdown(
    '''
    Este simulador representa o crescimento logístico. Os dados foram obtidos do seguinte artigo:
 
    [Experimental Studies on the Struggle for Existence, G. F. GAUSE, Journal of Experimental Biology 1932 9: 389-402](https://jeb.biologists.org/content/9/4/389).
    ''', mathjax=True
)

parametros = dcc.Markdown(
    '''
    * $r$: taxa de crescimento
    * $K$: capacidade suporte
    ''', mathjax=True
)

perguntas = dcc.Markdown(
    '''
    1. Estimar o valor da capacidade de suporte $K$ para Saccharomyces e Schizosaccharomices, visualmente a partir do gráfico.
    2. Estimar $$r$$ (por tentativa) para os dois casos.
    Dica:  $$r$$ entre $$0.10$$ e $$0.30$$ para Saccharomyces e $$r$$ entre $$0.0$$ e $$0.15$$ para Schizosaccharomyces.
    ''', mathjax=True
)

textos_descricao = html.Div(
        dbc.Accordion(
            [
                dbc.AccordionItem(
                    descricao, title="Descrição do modelo"
                ),
                dbc.AccordionItem(
                    parametros, title="Parâmetros do modelo"
                ),
                dbc.AccordionItem(
                    perguntas, title="Perguntas"
                ),
            ],
            start_collapsed=True,
        )
    )

ajuste_condicoes_iniciais = html.Div(
        [
            html.P("Parâmetros para Saccharomyces", className="card-header border-dark mb-3"),
            html.Div(
                [
                    dbc.Label(dcc.Markdown('''Taxa de crescimento ($$r$$):''', mathjax=True), html_for="r_sac"),
                    dcc.Input(placeholder='Digite um valor...', type='number', style={'width': '200px', 'maxWidth': '200px', 'minWidth': '200px'}, value='0', min='0', max='3', id="r_sac"),
                ],
                className="m-2",
            ),
            html.Div(
                [
                    dbc.Label(dcc.Markdown('''Capacidade suporte ($$K$$):''', mathjax=True), html_for="k_sac"),
                    dcc.Input(placeholder='Digite um valor...', type='number', style={'width': '200px', 'maxWidth': '200px', 'minWidth': '200px'}, value='3.14', min='1', max='15', id="k_sac"),
                ],
                className="m-2",
            ),

        ],
        className="card border-dark mb-3",
    )

ajuste_parametros = html.Div(
        [
            html.P("Parâmetros para Schizosaccharomyces", className="card-header border-dark mb-3"),
            html.Div(
                [
                    dbc.Label(dcc.Markdown('''Taxa de crescimento ($$r$$):''', mathjax=True), html_for="r_sch"),
                    dcc.Input(placeholder='Digite um valor...', type='number', style={'width': '200px', 'maxWidth': '200px', 'minWidth': '200px'}, value='0', min='0', max='3', id="r_sch"),
                ],
                className="m-2",
            ),
            html.Div(
                [
                    dbc.Label(dcc.Markdown('''Capacidade suporte ($$K$$):''', mathjax=True), html_for="k_sch"),
                    dcc.Input(placeholder='Digite um valor...', type='number', style={'width': '200px', 'maxWidth': '200px', 'minWidth': '200px'}, value='2.48', min='1', max='15', id="k_sch"),
                ],
                className="m-2",
            ),

        ],
        className="card border-dark mb-3",
    )

def ode_sys(state, t, r_sac, k_sac, r_sch, k_sch):
    sac, sch = state
    dsac_dt=r_sac*(1-(sac/k_sac))*sac
    dsch_dt=r_sch*(1-(sch/k_sch))*sch
    return [dsac_dt, dsch_dt]

@app.callback(Output('population_chart', 'figure'),
              [Input('r_sac', 'value'),
              Input('k_sac', 'value'),
              Input('r_sch', 'value'),
              Input('k_sch', 'value')])
def gera_grafico(r_sac, k_sac, r_sch, k_sch):
    try:
        r_sac = float(r_sac) if r_sac is not None else 0
    except ValueError:
        r_sac = 0.1 # Valor padrão em caso de erro de conversão

    try:
        k_sac = float(k_sac) if k_sac is not None else 3.14
    except ValueError:
        k_sac = 3.14

    try:
        r_sch = float(r_sch) if r_sch is not None else 0
    except ValueError:
        r_sch = 0.05

    try:
        k_sch = float(k_sch) if k_sch is not None else 2.48
    except ValueError:
        k_sch = 2.48

    t_begin = 0.
    t_end = 141.
    t_nsamples = 15000
    sac_0 = 0.4492
    sch_0 = 0.4636
    sac_real = [0.37,1.63,6.2,8.87,10.66,10.97,12.5,12.6,12.9,13.27,12.77,12.87,12.9,12.7]
    t_sac = [6,7.5,15,16,24,24,29,31.5,33,40,44,48,51.5,53]
    sch_real = [1.27,1,1.7,2.33,2.73,4.56,4.87,5.67,5.8,5.83]
    t_sch = [15,16,29,31.5,48,51.5,72,93,117,141]

    t_eval = np.linspace(t_begin, t_end, t_nsamples)
    sol = odeint(func=ode_sys,
                    y0=[sac_0,sch_0],
                    t=t_eval,
                    args=(r_sac, k_sac, r_sch, k_sch))
    sac, sch = sol.T
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_sac, y=sac_real, name='Saccharomyces',
                             mode='markers', marker=dict(color='blue',
                                         symbol='cross',  # Outro exemplo de símbolo
                                         size=4)
                            ) )
    fig.add_trace(go.Scatter(x=t_eval, y=sac, name='Ajuste Saccharomyces',
                             line=dict(color='blue', dash='dashdot', width=4)))
    fig.add_trace(go.Scatter(x=t_sch, y=sch_real, name='Schizosaccharomyces',
                             mode='markers', marker=dict(color='red',
                                         symbol='circle',  # Outro exemplo de símbolo
                                         size=4)
                            ) )
    fig.add_trace(go.Scatter(x=t_eval, y=sch, name='Ajuste Schizosaccharomyces',
                             line=dict(color='red', width=4)))
    fig.update_layout(title='Crescimento Logístico',
                       xaxis_title='Tempo (anos)',
                       yaxis_title='Número de indivíduos',
                       yaxis =dict(range=[0,16])
                       )
    return fig

app.layout = dbc.Container([
                cabecalho,
                dbc.Row([
                        dbc.Col([html.Div(ajuste_condicoes_iniciais),html.Div(ajuste_parametros)], width=3),
                        dbc.Col(html.Div(textos_descricao), width=3),
                        dbc.Col(dcc.Graph(id='population_chart', className="shadow-sm rounded-3 border-primary",
                                style={'height': '500px'}), width=6),
                ]),
              ], fluid=True),


if __name__ == '__main__':
    app.run(debug=False)
