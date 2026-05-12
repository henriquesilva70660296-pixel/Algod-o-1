import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import pytz

# 1. CONFIGURAÇÃO E ESTILO REFINADO
st_autorefresh(interval=30 * 1000, key="datarefresh")
st.set_page_config(page_title="Cotton Intel | Clean", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; border-radius: 10px; padding: 15px; border: 1px solid #30363d; }
    .card { background-color: #161b22; padding: 20px; border-radius: 10px; border-left: 5px solid #deff9a; margin-bottom: 20px; }
    div[data-testid="stExpander"] { border: none; background-color: #161b22; }
    </style>
    """, unsafe_allow_html=True)

# --- LOGICA DE DADOS (PRESERVADA) ---
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
    
    # --- HEADER ---
    st.title("🌱 Cotton Intelligence")
    
    # --- SEÇÃO 1: CARDS DE INTELIGÊNCIA (USDA & CLIMA) ---
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown(f"""<div class="card">
            <small>ESTOQUE MUNDIAL (USDA)</small><br>
            <b style="font-size:20px;">76.4M Fardos</b><br>
            <span style="color:#deff9a;">📉 Tendência de Baixa</span>
        </div>""", unsafe_allow_html=True)
        
    with col_info2:
        st.markdown(f"""<div class="card" style="border-left-color: #ff4b4b;">
            <small>RISCO CLIMÁTICO (TEXAS)</small><br>
            <b style="font-size:20px;">SECA NÍVEL D4</b><br>
            <span style="color:#ff4b4b;">⚠️ Risco de Safra Alto</span>
        </div>""", unsafe_allow_html=True)

    # --- SEÇÃO 2: MÉTRICAS DE MERCADO ---
    c1, c2, c3 = st.columns(3)
    c1.metric("COT. ALGODÃO", f"${preco_atual:.2f}")
    c2.metric("PETRÓLEO", f"${dados['Petroleo'].iloc[-1]:.1f}")
    c3.metric("DÓLAR (DXY)", f"{dados['Dolar'].iloc[-1]:.1f}")

    st.markdown("---")

    # --- SEÇÃO 3: IA DINÂMICA CENTRALIZADA ---
    prob = modelo.predict_proba(dados[features].tail(1))[0][1]
    
    st.subheader("Análise de Probabilidade")
    if prob > 0.65:
        st.success(f"🚀 **COMPRA FORTE:** O sistema detectou fundamentos sólidos ({prob*100:.1f}%)")
    elif prob > 0.45:
        st.warning(f"⚖️ **NEUTRO:** Mercado aguardando definições ({prob*100:.1f}%)")
    else:
        st.error(f"⚠️ **RISCO DE BAIXA:** Pressão vendedora detectada ({prob*100:.1f}%)")
    st.progress(prob)

    # --- SEÇÃO 4: EXECUÇÃO E ABAS ---
    st.sidebar.header("💰 Gestão de Conta")
    st.sidebar.metric("Saldo Líquido", f"${st.session_state.saldo:.2f}")
    if st.session_state.posicao > 0:
        st.sidebar.write(f"Posição: {st.session_state.posicao} Contratos")

    st.markdown("### Painel de Execução")
    qtd = st.number_input("Quantidade de Contratos:", min_value=1, value=100, step=50)
    col_b1, col_b2 = st.columns(2)
    
    with col_b1:
        if st.button("🟢 EXECUTAR COMPRA", use_container_width=True):
            st.session_state.saldo -= preco_atual * qtd
            st.session_state.posicao += qtd
            st.session_state.preco_entrada = preco_atual
            st.rerun()
    with col_b2:
        if st.button("🔴 FECHAR POSIÇÃO", use_container_width=True):
            if st.session_state.posicao > 0:
                lucro = (preco_atual - st.session_state.preco_entrada) * st.session_state.posicao
                st.session_state.saldo += preco_atual * st.session_state.posicao
                st.session_state.log_trades.append({"Data": datetime.now().strftime("%H:%M"), "Entrada": st.session_state.preco_entrada, "Saída": preco_atual, "P&L": round(lucro, 2)})
                st.session_state.historico_patrimonio.append(st.session_state.saldo)
                st.session_state.posicao = 0
                st.rerun()

    # ABAS ORGANIZADAS
    t1, t2, t3 = st.tabs(["📊 Gráfico Real", "📂 Log de Dados", "📈 Performance"])
    with t1:
        cor_graf = "#deff9a" if prob > 0.5 else "#ff4b4b"
        fig = go.Figure(go.Scatter(y=dados['Algodao'].tail(30), fill='tozeroy', line=dict(color=cor_graf, width=3)))
        fig.update_layout(height=300, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    with t2:
        if st.session_state.log_trades:
            st.table(pd.DataFrame(st.session_state.log_trades))
        else: st.info("Aguardando finalização de trades.")
        
    with t3:
        fig_eq = go.Figure(go.Scatter(y=st.session_state.historico_patrimonio, mode='lines+markers', line=dict(color='#deff9a')))
        fig_eq.update_layout(height=250, template="plotly_dark", title="Curva de Capital")
        st.plotly_chart(fig_eq, use_container_width=True)

except Exception as e:
    st.error(f"Erro: {e}")
