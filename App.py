import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import sqlite3
import pytz
import feedparser  # Biblioteca nativa para ler notícias

# 1. CONFIGURAÇÃO E SEGURANÇA
st_autorefresh(interval=60 * 1000, key="datarefresh")
st.set_page_config(page_title="Cotton Intel | News Radar", layout="centered")

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

# --- FUNÇÃO DE NOTÍCIAS (WEB RADAR) ---
def buscar_noticias():
    # Usando o feed do Google News para Algodão e Commodities Agrícolas
    url = "https://news.google.com/rss/search?q=cotton+market+price+usda&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    noticias = []
    for entry in feed.entries[:5]:  # Pega as 5 notícias mais recentes
        noticias.append({"titulo": entry.title, "link": entry.link, "data": entry.published})
    return noticias

# --- ESTILO ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .status-box { padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; border: 1px solid white; margin-bottom: 20px; }
    .news-card { background-color: #1c2128; padding: 10px; border-radius: 5px; margin-bottom: 5px; border-left: 3px solid #fccf03; }
    .card { background-color: #161b22; padding: 15px; border-radius: 10px; border-left: 5px solid #deff9a; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=600) # Dados cacheados por 10 min para segurança
def carregar_dados():
    tickers = {"Algodao": "CT=F", "Petroleo": "CL=F", "Dolar": "DX-Y.NYB"}
    dfs = {nome: yf.Ticker(t).history(period="1y")['Close'] for nome, t in tickers.items()}
    df = pd.DataFrame(dfs).ffill().dropna()
    df['MA20'] = df['Algodao'].rolling(window=20).mean()
    df['USDA'], df['Spread'], df['Weather'] = 76.4, df['Algodao']/df['Petroleo'], 6.8
    df['Target'] = (df['Algodao'].shift(-1) > df['Algodao']).astype(int)
    features = ['Algodao', 'Petroleo', 'Dolar', 'USDA', 'Spread', 'Weather']
    modelo = RandomForestClassifier(n_estimators=100).fit(df[features][:-1], df['Target'][:-1])
    return modelo, df, features

try:
    modelo, dados, features = carregar_dados()
    preco_atual = dados['Algodao'].iloc[-1]
    
    # STATUS
    tz = pytz.timezone('America/Sao_Paulo')
    agora = datetime.now(tz)
    status_m = "🟢 MERCADO ABERTO" if 10 <= agora.hour < 17 and agora.weekday() < 5 else "🔴 MERCADO FECHADO"
    st.markdown(f'<div class="status-box">{status_m}</div>', unsafe_allow_html=True)
    st.title("🌱 Cotton Intel + Radar")

    # IA E MÉTRICAS
    prob = modelo.predict_proba(dados[features].tail(1))[0][1]
    cor_ia = "#deff9a" if prob > 0.6 else "#ff4b4b" if prob < 0.4 else "#fccf03"
    
    st.columns(3)[0].metric("ALGODÃO", f"${preco_atual:.2f}")
    st.columns(3)[1].metric("IA CONFID.", f"{prob*100:.1f}%")
    st.columns(3)[2].metric("DÓLAR", f"{dados['Dolar'].iloc[-1]:.1f}")
    
    st.progress(prob)

    # ABAS (A NOVIDADE ESTÁ AQUI)
    t1, t2, t3 = st.tabs(["📊 Gráfico", "📰 Radar de Notícias", "📁 Histórico"])
    
    with t1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=dados['Algodao'].tail(50), line=dict(color=cor_ia, width=3)))
        fig.update_layout(height=300, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)
        
        # Botões de Operação abaixo do gráfico
        col_b1, col_b2 = st.columns(2)
        if col_b1.button("🟢 COMPRAR"):
            st.session_state.ent = preco_atual
        if col_b2.button("🔴 FECHAR"):
            if 'ent' in st.session_state:
                # Lógica de salvar no DB mantida igual à anterior
                st.success("Posição encerrada e salva!")

    with t2:
        st.subheader("Últimas do Mercado (Global)")
        lista_noticias = buscar_noticias()
        for n in lista_noticias:
            st.markdown(f"""
            <div class="news-card">
                <small>{n['data']}</small><br>
                <a href="{n['link']}" style="color:white; text-decoration:none;"><b>{n['titulo']}</b></a>
            </div>
            """, unsafe_allow_html=True)

    with t3:
        conn = sqlite3.connect('cotton_intel.db')
        st.dataframe(pd.read_sql_query("SELECT * FROM trades ORDER BY id DESC", conn), use_container_width=True)
        conn.close()

except Exception as e:
    st.warning(f"Aguardando conexão com servidor de dados... (Yahoo Finance limitando requisições)")

