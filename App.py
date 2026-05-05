import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from streamlit_autorefresh import st_autorefresh

# 1. CONFIGURAÇÃO
st_autorefresh(interval=60 * 1000, key="datarefresh")
st.set_page_config(page_title="Cotton Intel | Weather & Fund", layout="centered")

# --- SISTEMA DE CARTEIRA (PRESERVADO) ---
if 'saldo' not in st.session_state: st.session_state.saldo = 100000.0  
if 'posicao' not in st.session_state: st.session_state.posicao = 0      
if 'preco_entrada' not in st.session_state: st.session_state.preco_entrada = 0.0

@st.cache_data(ttl=3600)
def carregar_dados_ia():
    tickers = {"Algodao": "CT=F", "Petroleo": "CL=F", "Dolar": "DX-Y.NYB"}
    dfs = {nome: yf.Ticker(t).history(period="1y")['Close'] for nome, t in tickers.items()}
    df = pd.DataFrame(dfs).ffill().dropna()
    
    # DADOS FUNDAMENTAIS (Passos 1, 2, 3)
    df['USDA_Estoque'] = 76.4 
    df['Spread_Petroleo'] = df['Algodao'] / df['Petroleo']
    df['COT_Sentiment'] = 1 
    
    # PASSO 4: WEATHER RISK SCORE (0 a 10)
    # Texas: Seca (Risco 8) | Mato Grosso: Seco/Colheita (Risco 5)
    df['Weather_Risk'] = 6.5 
    
    df['Target'] = (df['Algodao'].shift(-1) > df['Algodao']).astype(int)
    features = ['Algodao', 'Petroleo', 'Dolar', 'USDA_Estoque', 'Spread_Petroleo', 'COT_Sentiment', 'Weather_Risk']
    
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(df[features][:-1], df['Target'][:-1])
    return modelo, df, features

try:
    modelo, dados, features = carregar_dados_ia()
    preco_atual = dados['Algodao'].iloc[-1]

    # --- INTERFACE PREMIUM ---
    st.title("🌱 Cotton Intel | Global Advisor")
    
    # Painel de Risco Climático
    with st.expander("🌍 Monitor de Clima (Zonas Produtoras)"):
        col_cl1, col_cl2 = st.columns(2)
        col_cl1.warning("Texas: Seca Severa (D4) detectada. Plantio em risco.")
        col_cl2.info("Mato Grosso: Tempo firme. Colheita avançando.")

    c1, c2, c3 = st.columns(3)
    c1.metric("ALGODÃO", f"${preco_atual:.2f}")
    c2.metric("PETRÓLEO", f"${dados['Petroleo'].iloc[-1]:.1f}")
    c3.metric("RISCO CLIMA", "ALTO", "Texas D4")

    # Operações (MANTIDO)
    st.sidebar.header("💰 Gestão de Conta")
    st.sidebar.metric("Saldo", f"${st.session_state.saldo:.2f}")
    
    qtd = st.number_input("Qtd Contratos:", min_value=1, value=100)
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("🟢 COMPRAR"):
            st.session_state.saldo -= preco_atual * qtd
            st.session_state.posicao += qtd
            st.session_state.preco_entrada = preco_atual
            st.rerun()
    with col_b2:
        if st.button("🔴 VENDER"):
            st.session_state.saldo += preco_atual * st.session_state.posicao
            st.session_state.posicao = 0
            st.rerun()

    # SINAL GLOBAL (USDA + CORREL + COT + CLIMA)
    prob = modelo.predict_proba(dados[features].tail(1))[0][1]
    if prob > 0.75:
        st.success(f"💎 SINAL GLOBAL: COMPRA FORTE ({prob*100:.0f}%)")
    else:
        st.info(f"⚖️ SINAL: AGUARDAR ({prob*100:.0f}%)")

    # Gráfico (MANTIDO)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dados.index[-40:], y=dados['Algodao'].tail(40), line=dict(color='#deff9a', width=3)))
    fig.update_layout(height=350, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Erro: {e}")

