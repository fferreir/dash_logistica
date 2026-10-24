# 📈 dash_logistica: Simulador do Modelo de Crescimento Logístico de Populações

Aplicação web interativa desenvolvida com **Dash**, **Plotly**, **Bootstrap** e **SciPy** para simulação, calibração e estudo de dinâmicas populacionais baseadas no clássico **Modelo Logístico de Verhulst-Pearl** e nos experimentos empíricos seminais de **G. F. Gause (1932)** com leveduras (*Saccharomyces cerevisiae* e *Schizosaccharomyces kephir*).

---

## 📌 Sumário
- [Visão Geral](#-visão-geral)
- [Formulação Matemática](#-formulação-matemática)
- [Parâmetros do Modelo](#-parâmetros-do-modelo)
- [Dados Experimentais de Gause (1932)](#-dados-experimentais-de-gause-1932)
- [Principais Funcionalidades](#-principais-funcionalidades)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Instalação e Execução](#-instalação-e-execução)
- [Roteiro de Estudo e Perguntas](#-roteiro-de-estudo-e-perguntas)
- [Licença](#-licença)

---

## 🔬 Visão Geral

O modelo de crescimento logístico é um dos pilares da ecologia teórica e da biologia matemática. Proposto originalmente por **Pierre François Verhulst** (1838) e reapresentado por **Pearl e Reed** (1920), descreve como uma população cresce em um ambiente com recursos finitos.

Em 1932, o biólogo soviético **Georgyi F. Gause** publicou um dos estudos mais importantes da história da ecologia experimental:
> **GAUSE, G. F.** *Experimental Studies on the Struggle for Existence: 1. Mixed Population of Two Species of Yeast.* **Journal of Experimental Biology**, v. 9, n. 4, p. 389–402, 1932.

Neste trabalho, Gause mediu com precisão o volume de biomassa acumulada por duas espécies de leveduras cultivadas isoladamente em condições padronizadas ao longo de até 141 horas, testando empiricamente as previsões do modelo logístico.

---

## 📐 Formulação Matemática

A dinâmica temporal da biomassa celular $N(t)$ é governada pela equação diferencial ordinária (EDO):

$$\frac{dN}{dt} = r N \left(1 - \frac{N}{K}\right)$$

Com a condição inicial $N(0) = N_0$, a solução analítica exata é dada por:

$$N(t) = \frac{K}{1 + \left(\frac{K - N_0}{N_0}\right) e^{-r t}}$$

### Propriedades Notáveis da Dinâmica Logística:
1. **Fase Exponencial Inicial ($N \ll K$):** Crescimento livre com tempo de duplicação:
   $$t_d = \frac{\ln(2)}{r} \approx \frac{0.6931}{r} \text{ horas}$$
2. **Ponto de Inflexão ($N = K/2$):** Ponto onde a taxa de crescimento absoluto atinge seu valor máximo:
   $$\left(\frac{dN}{dt}\right)_{\max} = \frac{r K}{4}$$
   O instante temporal em que a população atinge a inflexão é:
   $$t^* = \frac{1}{r} \ln\left(\frac{K - N_0}{N_0}\right)$$
3. **Fase Estacionária Assintótica ($t \to \infty$):** A taxa per capita de crescimento anula-se e a população estabiliza-se no patamar de capacidade de suporte $K$.

---

## 📊 Parâmetros do Modelo

| Parâmetro | Símbolo | Unidade | Significado Biológico |
| :--- | :---: | :---: | :--- |
| **Taxa intrínseca de crescimento** | $r$ | $\text{h}^{-1}$ | Potencial biótico reprodutivo per capita na ausência de limitações. |
| **Capacidade de suporte** | $K$ | $\text{vol}$ | Biomassa/densidade populacional máxima sustentável pelo meio. |
| **Inóculo inicial** | $N_0$ | $\text{vol}$ | Densidade populacional no instante inicial da inoculação ($t = 0$). |
| **Tempo de duplicação** | $t_d$ | $\text{horas}$ | Intervalo necessário para dobrar a biomassa na fase exponencial. |
| **Tempo de inflexão** | $t^*$ | $\text{horas}$ | Instante de máxima produtividade populacional ($N = K/2$). |
| **Aderência aos dados** | $R^2$ | Adimensional | Coeficiente de determinação entre o modelo e os dados de Gause. |
| **Erro médio** | $\text{RMSE}$ | $\text{vol}$ | Raiz do erro quadrático médio de ajuste. |

---

## 🧪 Dados Experimentais de Gause (1932)

* **Saccharomyces cerevisiae:** Levedura de brotamento assimétrico com multiplicação veloz. Atinge saturação em cerca de 40 a 50 horas com $K \approx 12.8 - 13.0 \text{ vol}$ e $r \approx 0.22 - 0.26 \text{ h}^{-1}$.
* **Schizosaccharomyces kephir:** Levedura de fissão binária simétrica. Apresenta ciclo celular mais longo, atingindo platô em $K \approx 5.8 \text{ vol}$ e $r \approx 0.057 - 0.060 \text{ h}^{-1}$ após mais de 100 horas.

---

## 🚀 Principais Funcionalidades

- **Painel de Controles com Badges:** Sliders com leitura em tempo real para $r$, $K$ e $N_0$ de ambas as espécies.
- **4 Cards de Métricas em Tempo Real (KPIs):**
  1. *Capacidade $K$ ($S. cerevisiae$)* e valor de referência de Gause.
  2. *Cinética ($S. cerevisiae$)* com tempo de duplicação $t_d$ e instante de inflexão.
  3. *Capacidade $K$ ($Sch. kephir$)* e valor de referência de Gause.
  4. *Qualidade do Ajuste ($R^2$ e RMSE)* atualizado dinamicamente contra os pontos reais.
- **Alternância de Variável Visualizada:** Alternância instantânea entre a curva de biomassa $N(t)$ e a taxa de variação instantânea $\frac{dN}{dt}$.
- **Elementos Didáticos no Gráfico:** Linhas de assíntota $K$, marcadores dos pontos de inflexão $K/2$ e pontos empíricos de Gause (1932).
- **Documentação Teórica Integrada:** Accordion com formulação matemática, metodologia experimental e roteiro pedagógico de perguntas.
- **Rodapé Institucional com Link para o GitHub:** Identificação do simulador e link direto para o repositório oficial de código aberto.

---

## 📁 Estrutura do Projeto

```text
dash_logistica/
├── app.py                     # Ponto de entrada da aplicação Dash e rotas
├── assets/
│   └── estilos.css            # Folha de estilos customizados (Hero, KPIs, Sliders)
├── pytest.ini                 # Configuração do executor de testes
├── requirements.txt           # Dependências Python do projeto
├── README.md                  # Documentação completa
├── LICENSE                    # Licença de uso
├── src/
│   ├── __init__.py            # Módulo de inicialização do pacote
│   ├── callbacks.py           # Callbacks reativos (gráficos, badges, simulação)
│   ├── layout.py              # Interface visual, header, controles, acordeom e rodapé
│   └── model.py               # EDOs, solução analítica, dados de Gause e métricas cinéticas
└── tests/
    ├── __init__.py            # Módulo de inicialização dos testes
    ├── test_app.py            # Testes de integração de interface, rodapé e layout
    └── test_model.py          # Testes unitários do modelo matemático e estabilidade
```

---

## 💻 Instalação e Execução

### 1. Clonar o repositório
```bash
git clone https://github.com/fferreir/dash_logistica.git
cd dash_logistica
```

### 2. Criar e ativar o ambiente virtual
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar as dependências
```bash
pip install -r requirements.txt
```

### 4. Executar os testes automatizados
```bash
pytest
```

### 5. Executar a aplicação
```bash
python app.py
```
O aplicativo estará disponível em: [http://localhost:8050](http://localhost:8050) (ou no subcaminho configurado `/dash_logistica/`).

---

## ❓ Roteiro de Estudo e Perguntas

1. **Identificação e Significado da Capacidade de Suporte ($K$):**
   Observe o comportamento assintótico dos dados experimentais nas horas finais ($t > 35\text{ h}$ para *Saccharomyces* e $t > 90\text{ h}$ para *Schizosaccharomyces*). Qual valor de $K$ melhor representa o platô estacionário de cada espécie? Por que a capacidade de suporte de *Saccharomyces* ($\approx 12.8$) é mais que o dobro da de *Schizosaccharomyces* ($\approx 5.8$), mesmo no mesmo meio de cultivo?
2. **Taxa Intrínseca ($r$) e Tempo de Duplicação ($t_d$):**
   Ajuste a taxa $r$ para ambas as espécies. Qual das duas leveduras apresenta maior velocidade de proliferação na fase inicial (exponencial)? Calcule o tempo de duplicação teórico ($t_d = \frac{\ln 2}{r}$) de cada espécie. Como o modo de divisão celular de cada organismo ajuda a justificar essa disparidade?
3. **Ponto de Inflexão e Taxa Máxima de Crescimento ($\frac{dN}{dt}$):**
   Habilite a visualização do ponto de inflexão ($N = K/2$) e alterne a escala do gráfico para *"Taxa de Crescimento dN/dt"*. Em que instante temporal ($t^*$) cada população atinge seu pico máximo de produtividade populacional? Por que a taxa instantânea de crescimento populacional começa a frear logo após a população atingir $K/2$?
4. **Sensibilidade ao Inóculo Inicial ($N_0$):**
   Varie a densidade inicial $N_0$ nos controles. O que acontece com a curva se o inóculo for extremamente reduzido ($N_0 \approx 0.1$)? E se você simular uma cultura iniciada com supersaturação ($N_0 > K$)? Como a equação logística responde matematicamente e qual seria o significado biológico?
5. **Aderência aos Dados ($R^2$, RMSE) e Limitações do Modelo:**
   Acompanhe os indicadores de qualidade de ajuste nos cards de métricas superiores. Qual conjunto de parâmetros ($r, K, N_0$) fornece o maior $R^2$ e o menor RMSE? Analise criticamente os primeiros pontos experimentais ($t < 15\text{ h}$): o modelo logístico prevê a fase de adaptação inicial (*lag phase*)? Quais as limitações de sistemas biológicos em batelada?

---

## 📄 Licença

Distribuído sob a licença GNU GPLv3. Consulte o arquivo [LICENSE](file:///home/fernando/Documentos/github/dash_logistica/LICENSE) para mais detalhes.
