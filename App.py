import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from streamlit_autorefresh import st_autorefresh

# 1. ATUALIZAÇÃO INSTANTÂNEA (30 segundos)
st_autorefresh(interval=30 * 1000, key="datarefresh")

st.set_page_config(page_title="Cotton Intel | Premium Pro", layout="centered")

# --- SISTEMA DE CARTEIRA ---
if 'saldo' not in st.session_state:
    st.session_state.saldo = 100000.0  
if 'posicao' not in st.session_state:
    st.session_state.posicao = 0      
if 'preco_entrada' not in st.session_state:
    st.session_state.preco_entrada = 0.0

# CSS Customizado
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #deff9a; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { 
        background-color: #161b22; border-radius: 5px; padding: 5px 15px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=0)
def carregar_dados():
    tickers = {"Algodao": "CT=F", "Petroleo": "CL=F", "Dolar": "DX-Y.NYB"}
    dfs = {nome: yf.Ticker(t).history(period="1y")['Close'] for nome, t in tickers.items()}
    df = pd.DataFrame(dfs).ffill().dropna()
    df['SMA_9'] = df['Algodao'].rolling(9).mean()
    df['SMA_21'] = df['Algodao'].rolling(21).mean()
    df['Target'] = (df['Algodao'].shift(-1) > df['Algodao']).astype(int)
    return df.dropna()

try:
    dados = carregar_dados()
    preco_atual = dados['Algodao'].iloc[-1]
    
    # 2. INTELIGÊNCIA ARTIFICIAL
    features = ['Algodao', 'Petroleo', 'Dolar', 'SMA_9', 'SMA_21']
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(dados[features][:-1], dados['Target'][:-1])
    prob = modelo.predict_proba(dados[features].tail(1))[0][1]

    # --- TÍTULO E MÉTRICAS ---
    st.title("🌱 Cotton Intelligence")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("ALGODÃO", f"${preco_atual:.2f}")
    c2.metric("PETRÓLEO", f"${dados['Petroleo'].iloc[-1]:.1f}")
    c3.metric("DÓLAR", f"{dados['Dolar'].iloc[-1]:.1f}")

    # --- PAINEL DE OPERAÇÕES ---
    st.sidebar.header("💰 Conta")
    st.sidebar.metric("Saldo", f"${st.session_state.saldo:.2f}")
    if st.session_state.posicao > 0:
        pnl = (preco_atual - st.session_state.preco_entrada) * st.session_state.posicao
        st.sidebar.metric("Lucro/Prejuízo", f"${pnl:.2f}", delta=f"{pnl:.2f}")

    st.markdown("### Execução Rápida")
    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        if st.button("🟢 COMPRAR (1000)"):
            custo = preco_atual * 1000
            if st.session_state.saldo >= custo:
                st.session_state.saldo -= custo
                st.session_state.posicao += 1000
                st.session_state.preco_entrada = preco_atual
                st.rerun()
    with col_b2:
        if st.button("🔴 VENDER (Zerar)"):
            if st.session_state.posicao > 0:
                st.session_state.saldo += preco_atual * st.session_state.posicao
                st.session_state.posicao = 0
                st.session_state.preco_entrada = 0
                st.rerun()
    with col_b3:
        st.metric("Entrada", f"${st.session_state.preco_entrada:.2f}")

    # --- SINAL IA ---
    if prob > 0.60:
        st.markdown(f"<h2 style='text-align: center; color: #deff9a;'>🚀 COMPRA FORTE ({prob*100:.0f}%)</h2>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h2 style='text-align: center; color: #64748b;'>📈 NEUTRO ({prob*100:.0f}%)</h2>", unsafe_allow_html=True)

    # --- GRÁFICOS ---
    tab1, tab2 = st.tabs(["📊 Tendência", "📈 Histórico"])
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dados.index[-20:], y=dados['Algodao'].tail(20), fill='tozeroy', line=dict(color='#deff9a')))
        if st.session_state.posicao > 0:
            fig.add_hline(y=st.session_state.preco_entrada, line_dash="dash", line_color="white")
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis=dict(side="right"))
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Erro: {e}")

