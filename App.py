import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import sqlite3
import pytz
import feedparser

# --- 1. CONFIGURAÇÃO E ESTABILIDADE ---
st_autorefresh(interval=60 * 1000, key="datarefresh")
st.set_page_config(page_title="Cotton Intel Pro", layout="wide")

# --- 2. BANCO DE DADOS ---
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

# --- 3. FONTES DE DADOS (Yahoo & RSS) ---
@st.cache_data(ttl=600)
def carregar_dados_mestre():
    # Preços
    tickers = {"Algodao": "CT=F", "Petroleo": "CL=F", "Dolar": "DX-Y.NYB"}
    dfs = {nome: yf.Ticker(t).history(period="1y")['Close'] for nome, t in tickers.items()}
    df = pd.DataFrame(dfs).ffill().dropna()
    
    # Indicadores e Normalização (Base 100)
    df['Volatilidade'] = df['Algodao'].diff().abs().rolling(14).mean()
    df_norm = (df / df.iloc[0]) * 100
    
    # IA
    df['Target'] = (df['Algodao'].shift(-1) > df['Algodao']).astype(int)
    features = ['Algodao', 'Petroleo', 'Dolar']
    modelo = RandomForestClassifier(n_estimators=100).fit(df[features][:-1], df['Target'][:-1])
    
    return modelo, df, df_norm, features

def buscar_noticias():
    url = "https://news.google.com/rss/search?q=cotton+market+price+usda&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    return [{"titulo": n.title, "link": n.link, "data": n.published} for n in feed.entries[:5]]

# --- 4. ESTILO VISUAL ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    .card { background-color: #161b22; padding: 15px; border-radius: 10px; border-left: 5px solid #deff9a; margin-bottom: 10px; }
    .status-bar { padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; margin-bottom: 20px; border: 1px solid rgba(255,255,255,0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- 5. LOGICA PRINCIPAL ---
try:
    modelo, df, df_norm, features = carregar_dados_mestre()
    preco_atual = df['Algodao'].iloc[-1]
    volat = df['Volatilidade'].iloc[-1]
    saldo_atual = get_saldo()

    # Cabeçalho de Mercado
    tz = pytz.timezone('America/Sao_Paulo')
    agora = datetime.now(tz)
    is_open = 10 <= agora.hour < 17 and agora.weekday() < 5
    status_txt = "🟢 MERCADO ABERTO" if is_open else "🔴 MERCADO FECHADO"
    st.markdown(f'<div class="status-bar">{status_txt} | {agora.strftime("%H:%M:%S")}</div>', unsafe_allow_html=True)

    # Painel Superior
    st.title("🌱 Cotton Intelligence Terminal")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ALGODÃO", f"${preco_atual:.2f}")
    c2.metric("PETRÓLEO", f"${df['Petroleo'].iloc[-1]:.1f}")
    c3.metric("DÓLAR", f"{df['Dolar'].iloc[-1]:.1f}")
    
    prob = modelo.predict_proba(df[features].tail(1))[0][1]
    cor_ia = "#deff9a" if prob > 0.6 else "#ff4b4b" if prob < 0.4 else "#fccf03"
    c4.metric("IA CONFIDENCE", f"{prob*100:.1f}%", delta_color="normal")

    # Abas do Terminal
    t_graf, t_corr, t_news, t_perf = st.tabs(["📊 Trade", "🔗 Macro", "📰 Notícias", "📈 Patrimônio"])

    with t_graf:
        col_main, col_risk = st.columns([2, 1])
        with col_main:
            fig = go.Figure(go.Scatter(y=df['Algodao'].tail(60), line=dict(color=cor_ia, width=3), fill='tozeroy'))
            fig.update_layout(template="plotly_dark", height=350, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)
        
        with col_risk:
            st.markdown('<div class="card"><b>GESTÃO DE RISCO</b></div>', unsafe_allow_html=True)
            risco_2 = saldo_atual * 0.02
            lote = int(risco_2 / ((volat * 2) * 100)) if volat > 0 else 100
            st.write(f"Saldo: ${saldo_atual:,.0f}")
            st.write(f"Risco Máx (2%): ${risco_2:.0f}")
            st.write(f"Lote Recomendado: **{lote} Ct**")
            
            qtd = st.number_input("Contratos", value=lote)
            if st.button("🟢 EXECUTAR COMPRA", use_container_width=True):
                st.session_state.ent = preco_atual
                st.session_state.q = qtd
                st.success("Ordem Executada!")
            
            if st.button("🔴 FECHAR POSIÇÃO", use_container_width=True):
                if 'ent' in st.session_state:
                    lucro = (preco_atual - st.session_state.ent) * st.session_state.q
                    conn = sqlite3.connect('cotton_intel.db')
                    conn.execute('UPDATE conta SET saldo = ?', (saldo_atual + lucro,))
                    conn.execute('INSERT INTO trades (data, tipo, entrada, saida, lucro) VALUES (?,?,?,?,?)',
                                 (datetime.now().strftime("%d/%m %H:%M"), "LONG", st.session_state.ent, preco_atual, lucro))
                    conn.commit()
                    conn.close()
                    del st.session_state.ent
                    st.rerun()

    with t_corr:
        st.subheader("Performance Comparativa (%)")
        fig_c = go.Figure()
        for c in df_norm.columns:
            fig_c.add_trace(go.Scatter(x=df_norm.index, y=df_norm[c], name=c))
        fig_c.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig_c, use_container_width=True)

    with t_news:
        for n in buscar_noticias():
            st.markdown(f'<div class="card"><small>{n["data"]}</small><br><a href="{n["link"]}">{n["titulo"]}</a></div>', unsafe_allow_html=True)

    with t_perf:
        conn = sqlite3.connect('cotton_intel.db')
        df_hist = pd.read_sql_query("SELECT lucro FROM trades", conn)
        conn.close()
        if not df_hist.empty:
            df_hist['equity'] = 100000 + df_hist['lucro'].cumsum()
            fig_e = go.Figure(go.Scatter(y=df_hist['equity'], fill='tozeroy', line=dict(color='#deff9a')))
            fig_e.update_layout(template="plotly_dark", title="Curva de Capital")
            st.plotly_chart(fig_e, use_container_width=True)
            st.dataframe(df_hist, use_container_width=True)
        else:
            st.info("Aguardando primeiro trade para gerar histórico.")

except Exception as e:
    st.warning("Reconectando aos servidores de mercado... Aguarde 60 segundos.")
