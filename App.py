import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import sqlite3
import pytz

# 1. CONFIGURAÇÃO
st_autorefresh(interval=30 * 1000, key="datarefresh")
st.set_page_config(page_title="Cotton Intel | Estável", layout="centered")

# --- BANCO DE DADOS (Para não perder seu saldo) ---
def init_db():
    conn = sqlite3.connect('cotton_intel.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS conta (id INTEGER PRIMARY KEY, saldo REAL)')
    c.execute('CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY, data TEXT, tipo TEXT, entrada REAL, saida REAL, lucro REAL)')
    c.execute('SELECT saldo FROM conta WHERE id = 1')
    if not c.fetchone():
        c.execute('INSERT INTO conta (id, saldo) VALUES (1, 100000.0)')
    conn.commit()
    conn.close()

init_db()

def get_saldo():
    conn = sqlite3.connect('cotton_intel.db')
    res = conn.execute('SELECT saldo FROM conta WHERE id = 1').fetchone()[0]
    conn.close()
    return res

# --- ESTILO ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .status-box { padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; border: 1px solid white; margin-bottom: 20px; }
    .card { background-color: #161b22; padding: 15px; border-radius: 10px; border-left: 5px solid #deff9a; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- LÓGICA DE MERCADO ---
def verificar_mercado():
    tz = pytz.timezone('America/Sao_Paulo')
    agora = datetime.now(tz)
    if agora.weekday() >= 5: return "🔴 MERCADO FECHADO", "#4a1010"
    return ("🟢 MERCADO ABERTO", "#104a10") if 10 <= agora.hour < 17 else ("🟡 AFTER-MARKET", "#4a4110")

@st.cache_data(ttl=60)
def carregar_dados():
    tickers = {"Algodao": "CT=F", "Petroleo": "CL=F", "Dolar": "DX-Y.NYB"}
    dfs = {nome: yf.Ticker(t).history(period="1y")['Close'] for nome, t in tickers.items()}
    df = pd.DataFrame(dfs).ffill().dropna()
    df['MA20'] = df['Algodao'].rolling(window=20).mean()
    df['USDA'], df['Spread'], df['COT'], df['Weather'] = 76.4, df['Algodao']/df['Petroleo'], 1, 6.8
    df['Target'] = (df['Algodao'].shift(-1) > df['Algodao']).astype(int)
    features = ['Algodao', 'Petroleo', 'Dolar', 'USDA', 'Spread', 'COT', 'Weather']
    modelo = RandomForestClassifier(n_estimators=100).fit(df[features][:-1], df['Target'][:-1])
    return modelo, df, features

try:
    modelo, dados, features = carregar_dados()
    preco_atual = dados['Algodao'].iloc[-1]
    msg_m, cor_m = verificar_mercado()
    
    # STATUS NO TOPO
    st.markdown(f'<div class="status-box" style="background-color: {cor_m};">{msg_m}</div>', unsafe_allow_html=True)
    st.title("🌱 Cotton Intelligence")

    # CARDS USDA E CLIMA
    c_i1, c_i2 = st.columns(2)
    with c_i1: st.markdown(f'<div class="card"><small>USDA ESTOQUE</small><br><b>76.4M</b></div>', unsafe_allow_html=True)
    with c_i2: st.markdown(f'<div class="card" style="border-left-color:orange"><small>CLIMA TEXAS</small><br><b>SECA D4</b></div>', unsafe_allow_html=True)

    # MÉTRICAS DE PREÇO
    col_p1, col_p2, col_p3 = st.columns(3)
    col_p1.metric("ALGODÃO", f"${preco_atual:.2f}")
    col_p2.metric("PETRÓLEO", f"${dados['Petroleo'].iloc[-1]:.1f}")
    col_p3.metric("DÓLAR", f"{dados['Dolar'].iloc[-1]:.1f}")

    # IA DINÂMICA
    prob = modelo.predict_proba(dados[features].tail(1))[0][1]
    cor_ia = "#deff9a" if prob > 0.6 else "#ff4b4b" if prob < 0.4 else "#fccf03"
    st.markdown(f"**Confiança da IA:** <span style='color:{cor_ia}'>{prob*100:.1f}%</span>", unsafe_allow_html=True)
    st.progress(prob)

    # GESTÃO E ORDENS
    saldo_atual = get_saldo()
    st.sidebar.metric("Saldo Atual", f"${saldo_atual:.2f}")
    
    qtd = st.number_input("Contratos:", 1, 1000, 100)
    c_b1, c_b2 = st.columns(2)
    with c_b1:
        if st.button("🟢 COMPRAR", use_container_width=True):
            st.session_state.ent = preco_atual
            st.session_state.q = qtd
            st.success("Comprado!")
    with c_b2:
        if st.button("🔴 FECHAR", use_container_width=True):
            if 'ent' in st.session_state:
                lucro = (preco_atual - st.session_state.ent) * st.session_state.q
                conn = sqlite3.connect('cotton_intel.db')
                conn.execute('UPDATE conta SET saldo = ?', (saldo_atual + lucro,))
                conn.execute('INSERT INTO trades (data, tipo, entrada, saida, lucro) VALUES (?,?,?,?,?)',
                             (datetime.now().strftime("%H:%M"), "COMPRA", st.session_state.ent, preco_atual, lucro))
                conn.commit()
                conn.close()
                del st.session_state.ent
                st.rerun()

    # GRÁFICO E HISTÓRICO
    t1, t2 = st.tabs(["📊 Gráfico", "📁 Histórico"])
    with t1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=dados['Algodao'].tail(50), line=dict(color=cor_ia, width=3)))
        fig.update_layout(height=300, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)
    with t2:
        conn = sqlite3.connect('cotton_intel.db')
        df_trades = pd.read_sql_query("SELECT * FROM trades ORDER BY id DESC", conn)
        st.dataframe(df_trades, use_container_width=True)
        conn.close()

except Exception as e:
    st.error(f"Erro: {e}")

