import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import pytz

# 1. CONFIGURAÇÃO E ATUALIZAÇÃO
st_autorefresh(interval=30 * 1000, key="datarefresh")
st.set_page_config(page_title="Cotton Intel | Ultimate Pro", layout="centered")

# --- SISTEMA DE CARTEIRA ---
if 'saldo' not in st.session_state: st.session_state.saldo = 100000.0  
if 'posicao' not in st.session_state: st.session_state.posicao = 0      
if 'preco_entrada' not in st.session_state: st.session_state.preco_entrada = 0.0

# CSS ESTILO PREMIUM
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #deff9a; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #161b22; border-radius: 5px; padding: 5px 15px; }
    </style>
    """, unsafe_allow_html=True)

def verificar_mercado():
    tz = pytz.timezone('America/Sao_Paulo')
    agora = datetime.now(tz)
    # Fins de semana: Fechado
    if agora.weekday() >= 5: return "🔴 MERCADO FECHADO (Fim de Semana)"
    # Horário simplificado do Algodão Futuros (NY) convertido para Brasília
    if 10 <= agora.hour < 17: return "🟢 MERCADO ABERTO (Pregão Principal)"
    else: return "🟡 MERCADO EM AFTER-MARKET / FECHADO"

@st.cache_data(ttl=0)
def carregar_dados_inteligentes():
    tickers = {"Algodao": "CT=F", "Petroleo": "CL=F", "Dolar": "DX-Y.NYB"}
    dfs = {nome: yf.Ticker(t).history(period="1y")['Close'] for nome, t in tickers.items()}
    df = pd.DataFrame(dfs).ffill().dropna()
    
    # INTEGRAÇÃO DOS 4 PASSOS NA IA
    df['USDA_Estoque'] = 76.4         # Passo 1: Fundamental
    df['Spread_Petroleo'] = df['Algodao'] / df['Petroleo'] # Passo 2: Correlação
    df['COT_Sentiment'] = 1           # Passo 3: Institucional
    df['Weather_Risk'] = 6.8          # Passo 4: Clima (Texas)
    
    df['Target'] = (df['Algodao'].shift(-1) > df['Algodao']).astype(int)
    features = ['Algodao', 'Petroleo', 'Dolar', 'USDA_Estoque', 'Spread_Petroleo', 'COT_Sentiment', 'Weather_Risk']
    
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(df[features][:-1], df['Target'][:-1])
    return modelo, df, features

try:
    modelo, dados, features = carregar_dados_inteligentes()
    preco_atual = dados['Algodao'].iloc[-1]
    status_mercado = verificar_mercado()
    
    # --- CABEÇALHO E STATUS ---
    st.title("🌱 Cotton Intelligence")
    st.markdown(f"**Status:** {status_mercado}")
    
    # Painel de Contexto Global
    with st.expander("🌍 Inteligência Global (USDA & Clima)"):
        c_i1, c_i2 = st.columns(2)
        c_i1.write("**USDA:** Estoques em queda (Bullish)")
        c_i2.write("**Clima:** Seca no Texas (Risco Médio/Alto)")

    # Métricas Principais
    c1, c2, c3 = st.columns(3)
    c1.metric("ALGODÃO", f"${preco_atual:.2f}")
    c2.metric("PETRÓLEO", f"${dados['Petroleo'].iloc[-1]:.1f}")
    c3.metric("DÓLAR", f"{dados['Dolar'].iloc[-1]:.1f}")

    # --- ÁREA DE TRADE ---
    st.sidebar.header("💰 Minha Conta")
    st.sidebar.metric("Saldo", f"${st.session_state.saldo:.2f}")
    if st.session_state.posicao > 0:
        pnl = (preco_atual - st.session_state.preco_entrada) * st.session_state.posicao
        st.sidebar.metric("Resultado", f"${pnl:.2f}", delta=f"{pnl:.2f}")

    st.markdown("### Painel de Ordens")
    qtd = st.number_input("Contratos:", min_value=1, value=100, step=50)
    
    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        if st.button("🟢 COMPRAR"):
            st.session_state.saldo -= preco_atual * qtd
            st.session_state.posicao += qtd
            st.session_state.preco_entrada = preco_atual
            st.rerun()
    with col_b2:
        if st.button("🔴 VENDER"):
            if st.session_state.posicao > 0:
                st.session_state.saldo += preco_atual * st.session_state.posicao
                st.session_state.posicao = 0
                st.rerun()
    with col_b3:
        st.metric("Entrada", f"${st.session_state.preco_entrada:.2f}")

    # SINAL GLOBAL
    prob = modelo.predict_proba(dados[features].tail(1))[0][1]
    st.markdown("---")
    if prob > 0.70:
        st.success(f"🚀 SINAL GLOBAL: COMPRA FORTE ({prob*100:.0f}%)")
    else:
        st.info(f"⚖️ SINAL: AGUARDAR / NEUTRO ({prob*100:.0f}%)")

    # GRÁFICO
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dados.index[-30:], y=dados['Algodao'].tail(30), fill='tozeroy', line=dict(color='#deff9a', width=3)))
    if st.session_state.posicao > 0:
        fig.add_hline(y=st.session_state.preco_entrada, line_dash="dash", line_color="white")
    fig.update_layout(height=350, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), yaxis=dict(side="right"))
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Erro na conexão: {e}")


