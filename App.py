import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import pytz

# 1. CONFIGURAÇÃO
st_autorefresh(interval=30 * 1000, key="datarefresh")
st.set_page_config(page_title="Cotton Intel | Dinâmico", layout="centered")

# ESTILO
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; border-radius: 10px; padding: 15px; border: 1px solid #30363d; }
    .card { background-color: #161b22; padding: 15px; border-radius: 10px; border-left: 5px solid #deff9a; margin-bottom: 10px; }
    .status-box { padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; margin-bottom: 20px; border: 1px solid white; }
    </style>
    """, unsafe_allow_html=True)

# STATUS DO MERCADO
def verificar_mercado():
    tz = pytz.timezone('America/Sao_Paulo')
    agora = datetime.now(tz)
    if agora.weekday() >= 5: return "🔴 MERCADO FECHADO (FDS)", "#4a1010"
    if 10 <= agora.hour < 17: return "🟢 MERCADO ABERTO (LIVE)", "#104a10"
    return "🟡 AFTER-MARKET / FECHADO", "#4a4110"

# LOGICA DE DADOS (PRESERVADA)
if 'saldo' not in st.session_state: st.session_state.saldo = 100000.0  
if 'posicao' not in st.session_state: st.session_state.posicao = 0      
if 'preco_entrada' not in st.session_state: st.session_state.preco_entrada = 0.0
if 'historico_patrimonio' not in st.session_state: st.session_state.historico_patrimonio = [100000.0]
if 'log_trades' not in st.session_state: st.session_state.log_trades = []

@st.cache_data(ttl=0)
def carregar_ia():
    tickers = {"Algodao": "CT=F", "Petroleo": "CL=F", "Dolar": "DX-Y.NYB"}
    dfs = {nome: yf.Ticker(t).history(period="1y")['Close'] for nome, t in tickers.items()}
    df = pd.DataFrame(dfs).ffill().dropna()
    df['USDA_Estoque'], df['Spread_Petroleo'], df['COT_Sentiment'], df['Weather_Risk'] = 76.4, df['Algodao']/df['Petroleo'], 1, 6.8
    df['Target'] = (df['Algodao'].shift(-1) > df['Algodao']).astype(int)
    features = ['Algodao', 'Petroleo', 'Dolar', 'USDA_Estoque', 'Spread_Petroleo', 'COT_Sentiment', 'Weather_Risk']
    modelo = RandomForestClassifier(n_estimators=100).fit(df[features][:-1], df['Target'][:-1])
    return modelo, df, features

try:
    modelo, dados, features = carregar_ia()
    preco_atual = dados['Algodao'].iloc[-1]
    msg_m, cor_m = verificar_mercado()
    
    # --- TOPO: STATUS ---
    st.markdown(f'<div class="status-box" style="background-color: {cor_m};">{msg_m}</div>', unsafe_allow_html=True)
    
    st.title("🌱 Cotton Intelligence")
    
    # --- CARDS INFO ---
    c_i1, c_i2 = st.columns(2)
    with c_i1:
        st.markdown(f'<div class="card"><small>USDA (ESTOQUE)</small><br><b>76.4M Fardos</b><br><span style="color:#deff9a;">📉 Baixo</span></div>', unsafe_allow_html=True)
    with c_i2:
        st.markdown(f'<div class="card" style="border-left-color: #ff4b4b;"><small>CLIMA (TEXAS)</small><br><b>SECA D4</b><br><span style="color:#ff4b4b;">⚠️ Risco Alto</span></div>', unsafe_allow_html=True)

    # --- MÉTRICAS ---
    c1, c2, c3 = st.columns(3)
    c1.metric("ALGODÃO", f"${preco_atual:.2f}")
    c2.metric("PETRÓLEO", f"${dados['Petroleo'].iloc[-1]:.1f}")
    c3.metric("DÓLAR", f"{dados['Dolar'].iloc[-1]:.1f}")

    # --- IA DINÂMICA COM CORES ---
    prob = modelo.predict_proba(dados[features].tail(1))[0][1]
    
    # Definindo a cor dinâmica
    if prob > 0.65:
        cor_sinal = "#deff9a" # Verde
        texto_sinal = "COMPRA FORTE"
    elif prob > 0.45:
        cor_sinal = "#fccf03" # Amarelo
        texto_sinal = "AGUARDAR / NEUTRO"
    else:
        cor_sinal = "#ff4b4b" # Vermelho
        texto_sinal = "RISCO DE BAIXA"

    st.markdown(f"**Confiança da IA:** <span style='color:{cor_sinal}; font-size:20px;'>{prob*100:.1f}%</span>", unsafe_allow_html=True)
    st.progress(prob)
    st.markdown(f"<p style='color:{cor_sinal}; font-weight:bold;'>SINAL: {texto_sinal}</p>", unsafe_allow_html=True)

    # --- PAINEL DE ORDENS ---
    st.sidebar.header("💰 Gestão")
    st.sidebar.metric("Saldo", f"${st.session_state.saldo:.2f}")
    
    st.markdown("### Painel de Ordens")
    qtd = st.number_input("Contratos:", min_value=1, value=100, step=50)
    col_b1, col_b2 = st.columns(2)
    
    with col_b1:
        if st.button("🟢 COMPRAR", use_container_width=True):
            st.session_state.saldo -= preco_atual * qtd
            st.session_state.posicao += qtd
            st.session_state.preco_entrada = preco_atual
            st.rerun()
    with col_b2:
        if st.button("🔴 FECHAR", use_container_width=True):
            if st.session_state.posicao > 0:
                lucro = (preco_atual - st.session_state.preco_entrada) * st.session_state.posicao
                st.session_state.saldo += preco_atual * st.session_state.posicao
                st.session_state.log_trades.append({"Data": datetime.now().strftime("%H:%M"), "Resultado": round(lucro, 2)})
                st.session_state.historico_patrimonio.append(st.session_state.saldo)
                st.session_state.posicao = 0
                st.rerun()

    # ABAS
    t1, t2, t3 = st.tabs(["📊 Gráfico", "📁 Log", "📈 Performance"])
    with t1:
        fig = go.Figure(go.Scatter(y=dados['Algodao'].tail(30), fill='tozeroy', line=dict(color=cor_sinal)))
        fig.update_layout(height=280, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)
    with t2:
        if st.session_state.log_trades: st.table(pd.DataFrame(st.session_state.log_trades))
    with t3:
        fig_eq = go.Figure(go.Scatter(y=st.session_state.historico_patrimonio, mode='lines+markers', line=dict(color='#deff9a')))
        fig_eq.update_layout(height=250, template="plotly_dark")
        st.plotly_chart(fig_eq, use_container_width=True)

except Exception as e:
    st.error(f"Erro: {e}")
